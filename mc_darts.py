import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import logging
from primitives import ALL_PRIMITIVES
from operations import OPS
from phylum import SEARCH_SPACE
from architect import Architect
from genotypes import Genotype, PRIMITIVES, NASNet, AmoebaNet, DARTS_V1, DARTS_V2
from utils import AvgrageMeter, accuracy
import numpy as np
import argparse
from types import SimpleNamespace

# Configure logging
logging.basicConfig(filename='error-file.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
# superNet models
class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, stride, op_names):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in op_names:
            op = OPS[primitive](C_in, stride).to(device)
            self._ops.append(op)
        self.C_in = C_in
        self.C_out = C_out
        if C_in != C_out:
            self.proj = nn.Sequential(
                nn.Conv2d(C_in, C_out, 1, bias=False),
                nn.BatchNorm2d(C_out)
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x, weights):
        weights = weights.view(-1)
        assert x.shape[1] == self.C_in, f"Expected {self.C_in} channels, got {x.shape[1]}"
        out = sum(w * op(x) for w, op in zip(weights, self._ops) if w > 1e-3)
        return self.proj(out)

class Cell(nn.Module):
    def __init__(self, C_prev, C_curr, reduction, cell_type, steps=4):
        super(Cell, self).__init__()
        self.cell_type = cell_type
        self.reduction = reduction
        self.steps = steps
        cell_config = SEARCH_SPACE.get_cell_config(cell_type)
        self.n_inputs = cell_config.get('n_inputs', 2)
        op_names = SEARCH_SPACE.get_operations(cell_type)

        self.preprocess = nn.ModuleList()
        for _ in range(self.n_inputs):
            if cell_type == 'MLP':
                self.preprocess.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(7),
                    nn.Flatten(),
                    nn.Linear(C_prev * 7 * 7, C_curr)
                ))
            else:
                self.preprocess.append(
                    nn.Sequential(
                        nn.Conv2d(C_prev, C_curr, 1, bias=False),
                        nn.BatchNorm2d(C_curr)
                    )
                )

        self._ops = nn.ModuleList()
        self._indices = []
        for i in range(self.steps):
            for j in range(self.n_inputs + i):
                stride = 2 if reduction and j < self.n_inputs and cell_type != 'MLP' else 1
                C_in = C_curr
                op = MixedOp(C_in, C_curr, stride, op_names)
                self._ops.append(op)
                self._indices.append(j)

    def forward(self, inputs, weights):
        states = [self.preprocess[i](inputs[i]) for i in range(self.n_inputs)]
        logging.info(f"Cell {self.cell_type}: Preprocessed states shapes: {[s.shape for s in states]}")
        offset = 0
        for i in range(self.steps):
            curr_states = [states[j] for j in range(self.n_inputs + i)]
            curr_weights = weights[offset:offset + len(curr_states)]
            s = sum(self._ops[offset + j](h, curr_weights[j]) for j, h in enumerate(curr_states))
            offset += len(curr_states)
            states.append(s)
        output = torch.cat(states[-self.steps:], dim=1)
        logging.info(f"Cell {self.cell_type}: Output shape: {output.shape}")
        return output

class MicroDARTS(nn.Module):
    def __init__(self, C=16, num_classes=10, layers=8, steps=4, genotype=None):
        super(MicroDARTS, self).__init__()
        self._layers = layers
        self._steps = steps
        self.cell_types = ['CNN', 'MLP', 'Fusion'] if genotype is None else ['CNN']
        self.cells_per_type = layers // len(self.cell_types) + 1 if genotype is None else layers

        C_curr = C
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
            nn.ReLU(inplace=True)
        )

        self.cells = nn.ModuleList()
        C_prev = C_curr
        layer_idx = 0

        if genotype is None:
            for cell_type in self.cell_types:
                cell_config = SEARCH_SPACE.get_cell_config(cell_type)
                reduction_cells = cell_config.get('reduction_cells', [2, 5])
                for i in range(self.cells_per_type):
                    if layer_idx >= layers:
                        break
                    reduction = layer_idx in reduction_cells
                    if cell_type == 'MLP':
                        C_curr = cell_config.get('hidden_sizes', [32])[0]
                    else:
                        C_curr = C  # Reset C_curr for CNN and Fusion to initial value
                    cell = Cell(C_prev, C_curr, reduction, cell_type, steps)
                    self.cells.append(cell)
                    num_ops = len(cell._ops)
                    self.register_parameter(f'alpha_{layer_idx}', nn.Parameter(torch.randn(num_ops)))
                    C_prev = C_curr * (self._steps if cell_type == 'Fusion' else 2)
                    if reduction and cell_type != 'MLP':
                        C_curr *= cell_config.get('channels', {}).get('increment', 2)
                    layer_idx += 1
                    logging.info(f"Layer {layer_idx}: C_prev={C_prev}, C_curr={C_curr}, Type={cell_type}, Reduction={reduction}")
        else:
            for i in range(layers):
                reduction = i in [2, 5]
                cell = Cell(C_prev, C_curr, reduction, 'CNN', steps)
                self.cells.append(cell)
                num_ops = len(cell._ops)
                self.register_parameter(f'alpha_{i}', nn.Parameter(torch.randn(num_ops)))
                C_prev = C_curr * 2
                if reduction:
                    C_curr *= 2

        self.global_pooling = nn.AdaptiveAvgPool2d(1) if self.cells[-1].cell_type != 'MLP' else nn.Identity()
        self.classifier = nn.Linear(C_prev, num_classes)
        self.genotype = genotype

    def forward(self, x):
        s0 = s1 = self.stem(x)
        logging.info(f"Stem output shape: {s1.shape}")
        for i, cell in enumerate(self.cells):
            if self.genotype and not cell.reduction:
                weights = self.genotype_weights(self.genotype.normal, i, cell)
            elif self.genotype and cell.reduction:
                weights = self.genotype_weights(self.genotype.reduce, i, cell)
            else:
                weights = F.softmax(getattr(self, f'alpha_{i}'), dim=0)
            s0, s1 = s1, cell([s0, s1], weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return F.cross_entropy(logits, target)

    def arch_parameters(self):
        return [getattr(self, f'alpha_{i}') for i in range(self._layers)]

    def new(self):
        model_new = MicroDARTS(C=16, num_classes=10, layers=self._layers, steps=self._steps, genotype=self.genotype).to(device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def genotype_weights(self, genotype_section, layer_idx, cell):
        weights = torch.zeros(len(cell._ops), device=device)
        offset = 0
        for node in range(self._steps):
            for j in range(self.n_inputs + node):
                if offset < len(genotype_section):
                    op_name, input_idx = genotype_section[offset]
                    if op_name in SEARCH_SPACE.get_operations('CNN'):
                        op_idx = SEARCH_SPACE.get_operations('CNN').index(op_name)
                        if input_idx == j:
                            weights[offset] = 1.0
                offset += 1
        return weights

def get_mnist_loader(batch_size=64, data_root="data"):
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    train_size = int(0.8 * len(train_set))
    valid_size = len(train_set) - train_size
    train_subset, valid_subset = torch.utils.data.random_split(train_set, [train_size, valid_size])
    test_set = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, valid_loader, test_loader

def train(model, train_loader, valid_loader, optimizer, criterion, architect, args):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    model.train()
    
    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)
    steps = min(len(train_loader), len(valid_loader))

    for step in range(steps):
        try:
            trn_images, trn_targets = next(train_iter)
            val_images, val_targets = next(valid_iter)
        except StopIteration:
            break

        trn_images, trn_targets = trn_images.to(device), trn_targets.to(device)
        val_images, val_targets = val_images.to(device), val_targets.to(device)

        if not model.genotype:
            architect.step(trn_images, trn_targets, val_images, val_targets, args.lr, optimizer, unrolled=True)

        optimizer.zero_grad()
        logits = model(trn_images)
        loss = criterion(logits, trn_targets)
        loss.backward()
        optimizer.step()

        prec1, _ = accuracy(logits, trn_targets, topk=(1, 5))
        n = trn_images.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)

    return top1.avg, objs.avg

def evaluate(model, test_loader):
    top1 = AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            prec1, _ = accuracy(logits, targets, topk=(1, 5))
            n = images.size(0)
            top1.update(prec1.item(), n)
    return top1.avg

def derive_genotype(model):
    if model.genotype:
        return model.genotype
    normal = []
    reduce = []
    normal_concat = list(range(2, 2 + model._steps))
    reduce_concat = list(range(2, 2 + model._steps))

    for i, cell in enumerate(model.cells):
        ## this step is Softmax Over Architecture Weights:
        weights = F.softmax(getattr(model, f'alpha_{i}'), dim=0)
        cell_ops = SEARCH_SPACE.get_operations(cell.cell_type)
        offset = 0
        for node in range(model._steps):
            curr_weights = weights[offset:offset + cell.n_inputs + node]
            if len(curr_weights) > 0:
                ## this step is to select top 2 operations with highest weights
                top2_indices = curr_weights.argsort(descending=True)[:min(2, len(curr_weights))]
                for idx in top2_indices:
                    ## Mapping to Operations: operation index is mapped to an actual operation from the search space.
                    op_idx = idx % len(cell_ops)
                    input_idx = idx // len(cell_ops)
                    op_name = cell_ops[op_idx]
                    if cell.reduction:
                        reduce.append((op_name, input_idx))
                    else:
                        normal.append((op_name, input_idx))
            offset += cell.n_inputs + node
    ## Architecture Finalization(discreatize): The final architecture is derived by concatenating the operations and the input indices.
    return Genotype(normal=normal[:8], normal_concat=normal_concat, reduce=reduce[:8], reduce_concat=reduce_concat)

def main():
    parser = argparse.ArgumentParser(description="DARTS on MNIST")
    parser.add_argument('--genotype', type=str, default=None, choices=['NASNet', 'AmoebaNet', 'DARTS_V1', 'DARTS_V2', None],
                        help="Use pre-defined genotype instead of searching")
    args = parser.parse_args()

    print("🚀 Running MicroDARTS on", device)
    training_config = SEARCH_SPACE.config['training']
    batch_size = training_config['batch_size']
    lr = training_config['learning_rate']
    weight_decay = training_config['weight_decay']
    epochs = training_config['epochs']['search']

    train_args = SimpleNamespace(
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
        arch_learning_rate=3e-4,
        arch_weight_decay=1e-3,
        cutout=False,
        cutout_length=16
    )

    train_loader, valid_loader, test_loader = get_mnist_loader(batch_size=batch_size)
    
    genotype_dict = {
        'NASNet': NASNet,
        'AmoebaNet': AmoebaNet,
        'DARTS_V1': DARTS_V1,
        'DARTS_V2': DARTS_V2
    }

    genotype = None
    if args.genotype:
        genotype = genotype_dict.get(args.genotype)
        if genotype is None:
            raise ValueError(f"Invalid genotype: {args.genotype}")
        print(f"Using pre-defined genotype: {args.genotype}")

    model = MicroDARTS(C=16, num_classes=10, layers=8, steps=4, genotype=genotype).to(device)
    optimizer = optim.SGD(model.parameters(), lr=train_args.lr, momentum=train_args.momentum, weight_decay=train_args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    architect = Architect(model, train_args) if not genotype else None

    print("✅ Model Initialized. Starting Training...\n")
    for epoch in range(epochs):
        print(f"🔄 Epoch {epoch + 1}/{epochs}: Training...")
        top1, loss = train(model, train_loader, valid_loader, optimizer, criterion, architect, train_args)
        test_acc = evaluate(model, test_loader)
        print(f"🎯 Epoch {epoch + 1}: Train Acc = {top1:.2f}%, Loss = {loss:.4f}, Test Acc = {test_acc:.2f}%")

    final_genotype = derive_genotype(model)
    print("\n🔥 Final Architecture:", final_genotype)
    
    from plot_genotype import plot
    plot(final_genotype.normal, "normal")
    plot(final_genotype.reduce, "reduction")

    if not args.genotype:
        print("\nComparison with Pre-defined Genotypes:")
        for name, geno in genotype_dict.items():
            print(f"{name}: {geno}")

    print("\n✅ Training Complete!")

if __name__ == "__main__":
    main()