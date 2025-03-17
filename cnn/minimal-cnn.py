import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

###################################################
# 1) Define minimal “search space” - NO batch_norm, NO skip_connect
###################################################
class SearchSpaceConfig:
    def __init__(self):
        # Only these ops remain
        self.CNN_operations = [
            "conv_1x1",
            "conv_3x3",
            "conv_5x5",
            "depthwise_conv_3x3",
            "max_pool_2x2",
            "avg_pool_2x2"
        ]
        self.cells = {
            "CNN": {
                "n_nodes": 4,
                "n_inputs": 2,
                "reduction_cells": [2, 5],
                "channels": {
                    "initial": 8,    # keep small for memory
                    "increment": 4,
                    "steps": [8, 16, 32, 64]
                },
                "dropout_rate": 0.3
            }
        }
        self.training = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0003,
            "epochs": {
                "search": 10,
                "train": 100
            }
        }
        self.architecture = {
            "init_channels": 8,
            "layers": 4,
            "auxiliary": {
                "enabled": False,
                "weight": 0.0
            },
            "drop_path_prob": 0.2
        }

    def get_operations(self, cell_type):
        if cell_type == "CNN":
            return self.CNN_operations
        return []

    def get_cell_config(self, cell_type):
        return self.cells.get(cell_type, {})

SEARCH_SPACE = SearchSpaceConfig()

###################################################
# 2) Define ops that produce consistent shapes
###################################################
def conv_1x1(C, stride):
    return nn.Conv2d(C, C, 1, stride=stride, padding=0, bias=False)

def conv_3x3(C, stride):
    return nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False)

def conv_5x5(C, stride):
    return nn.Conv2d(C, C, 5, stride=stride, padding=2, bias=False)

def depthwise_conv_3x3(C, stride):
    return nn.Conv2d(C, C, 3, stride=stride, padding=1, groups=C, bias=False)

OPS = {
    "conv_1x1": conv_1x1,
    "conv_3x3": conv_3x3,
    "conv_5x5": conv_5x5,
    "depthwise_conv_3x3": depthwise_conv_3x3,

    # For pooling ops, we do a 2x2 kernel, stride=2 as needed
    "max_pool_2x2": lambda C, stride: nn.MaxPool2d(kernel_size=2, stride=stride),
    "avg_pool_2x2": lambda C, stride: nn.AvgPool2d(kernel_size=2, stride=stride),
}

###################################################
# 3) MixedOp: each edge has multiple candidate ops
###################################################
class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        op_names = SEARCH_SPACE.get_operations("CNN")
        for name in op_names:
            op = OPS[name](C, stride)
            self._ops.append(op)

    def forward(self, x, weights):
        # weights shape: [n_ops]
        # each op must produce the same shape => sum is valid
        return sum(w * op(x) for w, op in zip(weights, self._ops))

###################################################
# 4) Cell: organizes multiple edges (MixedOps)
###################################################
class Cell(nn.Module):
    def __init__(self, C_in, C_out, reduction=False, steps=4):
        super().__init__()
        self.reduction = reduction
        self.steps = steps

        cell_config = SEARCH_SPACE.get_cell_config("CNN")
        self.n_inputs = cell_config.get("n_inputs", 2)
        self.n_ops = len(SEARCH_SPACE.get_operations("CNN"))

        # Preprocess each input
        self.preprocess = nn.ModuleList()
        for _ in range(self.n_inputs):
            self.preprocess.append(
                nn.Sequential(
                    nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),
                    nn.BatchNorm2d(C_out)  # just a preprocessor, not a candidate op
                )
            )

        self._ops = nn.ModuleList()
        self.edge_count = 0
        for i in range(self.steps):
            for j in range(self.n_inputs + i):
                stride = 2 if (reduction and j < self.n_inputs) else 1
                op = MixedOp(C_out, stride)
                self._ops.append(op)
                self.edge_count += 1

    def forward(self, inputs, alpha):
        states = []
        for i, x in enumerate(inputs):
            states.append(self.preprocess[i](x))

        offset_alpha = 0
        offset_ops = 0
        for i in range(self.steps):
            cur_states = states[: self.n_inputs + i]
            new_state = 0
            for s in cur_states:
                w_edge = alpha[offset_alpha : offset_alpha + self.n_ops]
                offset_alpha += self.n_ops
                new_state = new_state + self._ops[offset_ops](s, w_edge)
                offset_ops += 1
            states.append(new_state)

        return torch.cat(states[-self.n_inputs:], dim=1)

###################################################
# 5) The overall DARTS-like model
###################################################
class MicroDARTS(nn.Module):
    def __init__(self, init_channels=8, num_classes=10, layers=4, steps=4):
        super().__init__()
        self._layers = layers
        self._steps = steps
        self._num_classes = num_classes
        self.init_channels = init_channels

        self.stem = nn.Sequential(
            nn.Conv2d(1, init_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        cell_config = SEARCH_SPACE.get_cell_config("CNN")
        reduction_cells = cell_config.get("reduction_cells", [2, 5])

        self.cells = nn.ModuleList()
        C_prev = init_channels
        for i in range(layers):
            reduction = (i in reduction_cells)
            if reduction:
                C_curr = C_prev * 2
            else:
                C_curr = C_prev

            cell = Cell(C_prev, C_curr, reduction, steps)
            self.cells.append(cell)

            # alpha size = (#edges in cell) * (n_ops)
            E = cell.edge_count
            n_ops = cell.n_ops
            alpha_size = E * n_ops
            self.register_parameter(f"alpha_{i}", nn.Parameter(torch.randn(alpha_size)))

            n_inp = cell_config["n_inputs"]
            C_prev = C_curr * n_inp

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            alpha_i = getattr(self, f"alpha_{i}")
            weights = F.softmax(alpha_i, dim=0)
            s0, s1 = s1, cell([s0, s1], weights)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def arch_parameters(self):
        return [getattr(self, f"alpha_{i}") for i in range(self._layers)]

    def new(self):
        model_new = MicroDARTS(self.init_channels, self._num_classes, self._layers, self._steps).to(device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _loss(self, X, y):
        return F.cross_entropy(self.forward(X), y)

###################################################
# 6) Minimal Architect
###################################################
class Architect(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay
        )

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            # not implemented
            pass
        else:
            loss = self.model._loss(input_valid, target_valid)
            loss.backward()
        self.optimizer.step()

###################################################
# 7) Utility: data, training, eval
###################################################
def get_mnist_loader(batch_size=32, data_root="data"):
    # Transform with NO rotation or flip => ensures shape is always 28x28
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_set = torchvision.datasets.MNIST(
        root=data_root, train=True, download=False, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root=data_root, train=False, download=False, transform=transform
    )

    train_size = int(0.8 * len(full_train_set))
    valid_size = len(full_train_set) - train_size
    train_subset, valid_subset = random_split(full_train_set, [train_size, valid_size])

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, valid_loader, test_loader

class AvgrageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

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

        # Architecture step
        architect.step(trn_images, trn_targets, val_images, val_targets, args.lr, optimizer, unrolled=False)

        # Network step
        optimizer.zero_grad()
        logits = model(trn_images)
        loss = criterion(logits, trn_targets)
        loss.backward()
        optimizer.step()

        prec1, = accuracy(logits, trn_targets, topk=(1,))
        n = trn_images.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)

    print(f"Epoch {epoch+1} => train_acc={top1.avg:.2f}%, loss={objs.avg:.4f}")
    return top1.avg, objs.avg

def evaluate(model, loader, name="Test"):
    top1 = AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            prec1, = accuracy(logits, targets, topk=(1,))
            top1.update(prec1.item(), images.size(0))
    print(f"{name} Accuracy: {top1.avg:.2f}%")
    return top1.avg

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

def derive_genotype(model):
    normal = []
    reduce = []
    normal_concat = list(range(2, 2 + model._steps))
    reduce_concat = list(range(2, 2 + model._steps))
    for i, cell in enumerate(model.cells):
        alpha_i = getattr(model, f"alpha_{i}")
        weights = F.softmax(alpha_i, dim=0)
        if cell.reduction:
            reduce.append(("<some_op>", 0))
        else:
            normal.append(("<some_op>", 0))
    return Genotype(normal=normal, normal_concat=normal_concat,
                    reduce=reduce, reduce_concat=reduce_concat)

def main():
    import argparse
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description="DARTS on MNIST - No random transforms => consistent 28x28")
    parser.add_argument("--epochs", type=int, default=SEARCH_SPACE.training["epochs"]["search"])
    args = parser.parse_args()

    train_loader, valid_loader, test_loader = get_mnist_loader(batch_size=SEARCH_SPACE.training["batch_size"])
    model = MicroDARTS(
        init_channels=SEARCH_SPACE.architecture["init_channels"],
        num_classes=10,
        layers=SEARCH_SPACE.architecture["layers"],
        steps=4
    ).to(device)

    arch_args = SimpleNamespace(
        lr=SEARCH_SPACE.training["learning_rate"],
        momentum=0.9,
        weight_decay=SEARCH_SPACE.training["weight_decay"],
        arch_learning_rate=3e-4,
        arch_weight_decay=1e-3
    )
    architect = Architect(model, arch_args)
    optimizer = optim.SGD(
        model.parameters(),
        lr=arch_args.lr,
        momentum=arch_args.momentum,
        weight_decay=arch_args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_acc, train_loss = train(model, train_loader, valid_loader,
                                      optimizer, criterion, architect, arch_args, epoch)
        val_acc = evaluate(model, valid_loader, name="Valid")
        test_acc = evaluate(model, test_loader,  name="Test")

    genotype = derive_genotype(model)
    print("\nFinal Genotype:", genotype)
    print("Done.")

if __name__ == "__main__":
    main()
