# mc_darts.py
# Bilevel search logic and main flow specialized for CNN-only on MNIST.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import logging
import argparse
from torch.utils.data import random_split
from phylum import SEARCH_SPACE
from model_search import MixedOp
from operations import OPS
from genotypes import Genotype
from architect import Architect
from utils import AvgrageMeter, accuracy
from types import SimpleNamespace

logging.basicConfig(filename='error-file.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

def get_mnist_loader(batch_size=64, data_root="data"):
    # Ensure MNIST is already downloaded
    assert os.path.exists(os.path.join(data_root, "MNIST/raw")), (
        "❌ MNIST data not found in 'data/MNIST/raw' -- please ensure it's pre-downloaded."
    )

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the entire official training set
    full_train_set = torchvision.datasets.MNIST(
        root=data_root, train=True, download=False, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root=data_root, train=False, download=False, transform=transform
    )

    # Split the full training set into 80% train / 20% valid
    train_size = int(0.8 * len(full_train_set))
    valid_size = len(full_train_set) - train_size
    train_subset, valid_subset = random_split(full_train_set, [train_size, valid_size])

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_subset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, valid_loader, test_loader

class Cell(nn.Module):
    """
    A basic CNN cell with multiple 'MixedOp's. 
    """
    def __init__(self, C_in, C_out, reduction=False, steps=4):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.steps = steps
        cell_config = SEARCH_SPACE.get_cell_config('CNN')
        self.n_inputs = cell_config.get('n_inputs', 2)

        self.preprocess = nn.ModuleList()
        for _ in range(self.n_inputs):
            # 1x1 conv to match channels
            self.preprocess.append(
                nn.Sequential(
                    nn.Conv2d(C_in, C_out, 1, bias=False),
                    nn.BatchNorm2d(C_out)
                )
            )

        self._ops = nn.ModuleList()
        self._indices = []
        for i in range(self.steps):
            for j in range(self.n_inputs + i):
                stride = 2 if reduction and j < self.n_inputs else 1
                op = MixedOp(C_out, stride)
                self._ops.append(op)
                self._indices.append(j)

    def forward(self, inputs, weights):
        # Preprocess each input
        states = []
        for i, inp in enumerate(inputs):
            states.append(self.preprocess[i](inp))

        offset = 0
        for i in range(self.steps):
            cur_states = states[: self.n_inputs + i]
            cur_weights = weights[offset : offset + len(cur_states)]
            s = sum(self._ops[offset + j](h, cur_weights[j]) for j, h in enumerate(cur_states))
            offset += len(cur_states)
            states.append(s)

        return torch.cat(states[-self.n_inputs:], dim=1)

class MicroDARTS(nn.Module):
    """
    A simplified DARTS-like network that uses only CNN cells on MNIST.
    """
    def __init__(self, init_channels=16, num_classes=10, layers=8, steps=4):
        super(MicroDARTS, self).__init__()
        self._layers = layers
        self._steps = steps
        self._num_classes = num_classes
        self.init_channels = init_channels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, init_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        # Build cells
        self.cells = nn.ModuleList()
        C_prev = init_channels
        cell_config = SEARCH_SPACE.get_cell_config('CNN')
        reduction_cells = cell_config.get('reduction_cells', [2, 5])

        for i in range(layers):
            reduction = i in reduction_cells
            # Output channels remain the same unless we reduce
            C_curr = C_prev
            if reduction:
                # If reduction, effectively double the next channels
                C_curr *= 2
            cell = Cell(C_prev, C_curr, reduction, steps)
            self.cells.append(cell)
            num_ops = len(cell._ops)
            self.register_parameter(f'alpha_{i}', nn.Parameter(torch.randn(num_ops)))
            # For the next cell, the channel dimension is cat of the last n_inputs states
            # each of dimension C_curr
            C_prev = C_curr * cell_config['n_inputs']

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            weights = F.softmax(getattr(self, f'alpha_{i}'), dim=0)
            s0, s1 = s1, cell([s0, s1], weights)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def arch_parameters(self):
        return [getattr(self, f'alpha_{i}') for i in range(self._layers)]

    def new(self):
        model_new = MicroDARTS(
            init_channels=self.init_channels,
            num_classes=self._num_classes,
            layers=self._layers,
            steps=self._steps
        ).to(device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _loss(self, x, y):
        logits = self.forward(x)
        return F.cross_entropy(logits, y)

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

        # Architecture step (bi-level optimization)
        architect.step(trn_images, trn_targets, val_images, val_targets, args.lr, optimizer, unrolled=True)

        # Network step
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

def evaluate(model, loader):
    top1 = AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            prec1, _ = accuracy(logits, targets, topk=(1, 5))
            n = images.size(0)
            top1.update(prec1.item(), n)
    return top1.avg

def derive_genotype(model):
    """
    Convert the learned architecture (softmax over alpha weights)
    into a discrete Genotype with normal and reduce structures.
    This is a simplified approach, since we are CNN-only.
    """
    normal = []
    reduce = []
    normal_concat = list(range(2, 2 + model._steps))
    reduce_concat = list(range(2, 2 + model._steps))
    
    for i, cell in enumerate(model.cells):
        weights = F.softmax(getattr(model, f'alpha_{i}'), dim=0)
        # For demonstration, pick top 2 Weighted ops (like standard DARTS)
        offset = 0
        n_inputs = SEARCH_SPACE.get_cell_config('CNN')['n_inputs']
        steps = cell.steps
        is_reduce = cell.reduction
        local_list = reduce if is_reduce else normal

        for node in range(steps):
            n_choices = n_inputs + node
            chunk_weights = weights[offset : offset + n_choices]
            top2 = chunk_weights.argsort(descending=True)[:2]
            offset += n_choices
            for idx in top2:
                op_idx = idx.item()  # which input
                op_weight = chunk_weights[op_idx]
                # We won't map to actual op name index-by-index, so let's just store (\"op\", input_idx).
                # For a real approach, you'd do a separate reference. We'll approximate:
                local_list.append(("<chosen_op>", op_idx))

    return Genotype(
        normal=normal,
        normal_concat=normal_concat,
        reduce=reduce,
        reduce_concat=reduce_concat
    )

def main():
    parser = argparse.ArgumentParser(description="DARTS on MNIST (CNN only)")
    parser.add_argument('--genotype', type=str, default=None, help="Unused in this minimal variant.")
    args = parser.parse_args()

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
        arch_weight_decay=1e-3
    )

    train_loader, valid_loader, test_loader = get_mnist_loader(batch_size)
    
    model = MicroDARTS(
        init_channels=SEARCH_SPACE.config['architecture']['init_channels'],
        num_classes=10,
        layers=SEARCH_SPACE.config['architecture']['layers'],
        steps=4
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=train_args.lr, momentum=train_args.momentum, weight_decay=train_args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    architect = Architect(model, train_args)

    print("✅ Model Initialized. Starting Training...")
    for epoch in range(epochs):
        top1, avg_loss = train(model, train_loader, valid_loader, optimizer, criterion, architect, train_args)
        test_acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}/{epochs} => Train Acc: {top1:.2f}%, Loss: {avg_loss:.4f}, Test Acc: {test_acc:.2f}%")

    final_genotype = derive_genotype(model)
    print("\n🔥 Final Architecture:", final_genotype)
    print("✅ Training Complete!")
    # Optionally, you can plot or save the genotype here

if __name__ == "__main__":
    main()
