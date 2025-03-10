# mc_darts.py
# Bilevel search logic and main flow specialized for CNN-only on MNIST.
# Modified to reduce memory usage, reduce the chance of system shutdown, and allow checkpointing.

import torch, os, onnx
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

CHECKPOINT_PATH = "mnist_darts_checkpoint.pth"

def save_checkpoint(model, optimizer, epoch, path=CHECKPOINT_PATH):
    """Save current model/optimizer state so we can resume later if needed."""
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, path)

def load_checkpoint(path, model, optimizer):
    """Load saved state if it exists, returning the next epoch index to continue from."""
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        return checkpoint['epoch'] + 1
    return 0

def get_mnist_loader(batch_size=32, data_root="data"):
    # If you are sure you have MNIST downloaded, keep the assertion. Otherwise, remove it.
    assert os.path.exists(os.path.join(data_root, "MNIST/raw")), (
        "❌ MNIST data not found in 'data/MNIST/raw' -- please ensure it's pre-downloaded."
    )

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_set = torchvision.datasets.MNIST(
        root=data_root, train=True, download=False, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root=data_root, train=False, download=False, transform=transform
    )

    # Split into 80% train, 20% valid
    train_size = int(0.8 * len(full_train_set))
    valid_size = len(full_train_set) - train_size
    train_subset, valid_subset = random_split(full_train_set, [train_size, valid_size])

    # DataLoaders with fewer workers to reduce CPU overhead
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_subset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    return train_loader, valid_loader, test_loader

# mc_darts.py
class Cell(nn.Module):
    def __init__(self, input_channels, C_out, reduction=False, steps=4):
        super().__init__()
        self.reduction = reduction
        self.steps = steps
        self.n_inputs = 2

        # Preprocessing layers use stride=2 for reduction cells
        self.preprocess = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels[0], C_out, 1,
                          stride=2 if reduction else 1,  # <-- Add stride
                          bias=False),
                nn.BatchNorm2d(C_out)
            ),
            nn.Sequential(
                nn.Conv2d(input_channels[1], C_out, 1,
                          stride=2 if reduction else 1,  # <-- Add stride
                          bias=False),
                nn.BatchNorm2d(C_out)
            )
        ])

        # Build mixed ops with uniform stride
        self._ops = nn.ModuleList()
        for i in range(steps):
            for j in range(self.n_inputs + i):
                op_stride = 2 if reduction else 1  # All ops in cell share stride
                self._ops.append(MixedOp(C_out, op_stride))

    def forward(self, inputs, alphas):
        # alphas shape: [num_edges, n_ops]
        weights = F.softmax(alphas, dim=-1)

        states = []
        for i, inp in enumerate(inputs):
            processed = self.preprocess[i](inp)
            #print(f"Preprocessed input {i}: {processed.shape}")  # Debug shape
            states.append(processed)

        offset = 0
        for i in range(self.steps):
            cur_states = states[: self.n_inputs + i]
            for j, h in enumerate(cur_states):
                edge_idx = offset + j
                op_weights = weights[edge_idx]
                s = self._ops[edge_idx](h, op_weights)
                #print(f"Op {edge_idx} output: {s.shape}")  # Debug shape
                states.append(s)
            offset += len(cur_states)

        # Ensure all states have the same spatial dimensions before concatenation
        target_shape = states[-1].shape[2:]
        for i in range(len(states)):
            if states[i].shape[2:] != target_shape:
                states[i] = F.interpolate(states[i], size=target_shape)

        output = torch.cat(states[-self.n_inputs:], dim=1)
        #print(f"Final cell output: {output.shape}")  # Debug shape
        return output

# mc_darts.py
class MicroDARTS(nn.Module):
    def __init__(self, init_channels=16, num_classes=10, layers=4, steps=4):
        super().__init__()
        self._layers = layers
        self._num_classes = num_classes
        self._steps = steps  # Add this line
        self.init_channels = init_channels
        cell_config = SEARCH_SPACE.get_cell_config('CNN')
        self.n_ops = len(SEARCH_SPACE.get_operations('CNN'))

        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv2d(1, init_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU()
        )

        # Track channels
        s0_ch = s1_ch = init_channels
        self.cells = nn.ModuleList()
        for i in range(layers):
            reduction = i in cell_config['reduction_cells']
            C_curr = s1_ch * 2 if reduction else s1_ch

            cell = Cell([s0_ch, s1_ch], C_curr, reduction)
            num_edges = len(cell._ops)
            self.register_parameter(
                f'alpha_{i}',
                nn.Parameter(1e-3 * torch.randn(num_edges, self.n_ops))
            )
            self.cells.append(cell)

            # Update channels: output is concatenation of n_inputs states
            s0_ch, s1_ch = s1_ch, C_curr * cell.n_inputs  # <-- Multiply by n_inputs

        # Classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(s1_ch, num_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            alphas = getattr(self, f'alpha_{i}')  # [num_edges, n_ops]
            s0, s1 = s1, cell([s0, s1], alphas)
        out = self.global_pooling(s1)
        return self.classifier(out.view(out.size(0), -1))

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

def train(model, train_loader, valid_loader, optimizer, criterion, architect, args, epoch):
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
        architect.step(trn_images, trn_targets, val_images, val_targets, args.lr, optimizer, unrolled=False)

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

    print(f"Epoch {epoch+1} => Train Acc: {top1.avg:.2f}%, Loss: {objs.avg:.4f}")
    return top1.avg, objs.avg

def evaluate(model, loader, split_name="Test"):
    top1 = AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            prec1, _ = accuracy(logits, targets, topk=(1, 5))
            n = images.size(0)
            top1.update(prec1.item(), n)
    print(f"{split_name} Acc: {top1.avg:.2f}%")
    return top1.avg

def derive_genotype(model):
    """
    Convert the learned architecture (softmax over alpha weights)
    into a discrete Genotype with normal and reduce structures.
    """
    normal = []
    reduce = []
    cell_config = SEARCH_SPACE.get_cell_config('CNN')
    n_inputs = cell_config['n_inputs']
    steps = cell_config['n_nodes']  # Assuming 'n_nodes' is defined in search_space.json

    for i, cell in enumerate(model.cells):
        alphas = getattr(model, f'alpha_{i}')  # Shape: [num_edges, n_ops]
        is_reduce = cell.reduction
        local_list = reduce if is_reduce else normal

        # Iterate over edges and select top operations
        for edge_idx in range(alphas.shape[0]):
            edge_weights = F.softmax(alphas[edge_idx], dim=-1)  # [n_ops]
            top_op_idx = edge_weights.argmax().item()  # Get scalar index
            op_name = SEARCH_SPACE.get_operations('CNN')[top_op_idx]
            local_list.append((op_name, edge_idx % n_inputs))  # Assuming 2 inputs

    return Genotype(
        normal=normal,
        normal_concat=list(range(2, 2 + steps)),
        reduce=reduce,
        reduce_concat=list(range(2, 2 + steps))
    )

def main():
    parser = argparse.ArgumentParser(description="DARTS on MNIST (CNN only) - Memory-Optimized")
    parser.add_argument('--genotype', type=str, default=None, help="Unused in this minimal variant.")
    parser.add_argument('--resume', action='store_true', help="Resume from checkpoint if available.")
    args = parser.parse_args()

    training_config = SEARCH_SPACE.config['training']
    batch_size = 32  # Reduced to minimize memory usage
    lr = training_config['learning_rate']
    weight_decay = training_config['weight_decay']
    epochs = 10  # Fewer search epochs if memory is tight

    # Simplified training config
    train_args = SimpleNamespace(
        lr=0.025,
        momentum=0.9,
        weight_decay=3e-4,
        arch_learning_rate=3e-4,
        arch_weight_decay=1e-3,
        unrolled=False
    )

    # Get data loaders
    train_loader, valid_loader, test_loader = get_mnist_loader(batch_size=batch_size)

    # Verify data dimensions

    # Build model with smaller config
    model = MicroDARTS(
        init_channels=8,  # reduced
        num_classes=10,
        layers=4,        # reduced
        steps=4
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=train_args.lr, momentum=train_args.momentum, weight_decay=train_args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    architect = Architect(model, train_args)

    # Optionally resume from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(CHECKPOINT_PATH, model, optimizer)
        print(f"Resuming from epoch {start_epoch}...")
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state'])
            print(f"✅ Loaded checkpoint from {CHECKPOINT_PATH}")
            # Save to ONNX
            save_as_onnx(model, output_path="d_mnist_model.onnx")
            #save_as_onnx(model)
        else:
            print(f"❌ Checkpoint file {CHECKPOINT_PATH} not found")

    print("✅ Model Initialized. Starting Training...")
    for epoch in range(start_epoch, epochs):
        train_acc, train_loss = train(model, train_loader, valid_loader, optimizer, criterion, architect, train_args, epoch)
        valid_acc = evaluate(model, valid_loader, split_name="Valid")
        test_acc = evaluate(model, test_loader, split_name="Test")

        # Save checkpoint every epoch
        save_checkpoint(model, optimizer, epoch)

    final_genotype = derive_genotype(model)
    print("\n🔥 Final Architecture:", final_genotype)
    print("✅ Training Complete!")
    save_as_onnx(model)

# Save final model to ONNX format
def save_as_onnx(model, input_shape=(1, 1, 28, 28), output_path="daarts_model.onnx"):
    # Create dummy input
    dummy_input = torch.randn(*input_shape).cpu()
    model.cpu().eval()
    # Export the model
    try:
        torch.onnx.export(
            model,  # Model to export
            dummy_input,  # Example input
            output_path,  # Output path
            export_params=True,  # Store trained parameters
            opset_version=11,  # ONNX opset version
            do_constant_folding=True,  # Optimize constants
            input_names=["input"],  # Input name
            output_names=["output"],  # Output name
            dynamic_axes={  # Dynamic axes (batch dimension)
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
                dynamo=True
        )
        print(f"✅ Model saved as ONNX to {output_path}")
    except Exception as e:
        print(f"❌ Error during ONNX export: {e}")

if __name__ == "__main__":
    main()
