import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import os
import torchvision.transforms as transforms
import logging
from primitives import ALL_PRIMITIVES
from operations import OPS
from phylum import SEARCH_SPACE

# Clear the log file
with open('error-file.log', 'w'):
    pass

# Configure logging
logging.basicConfig(filename='error-file.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(message)s')

# Set device for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, stride, op_names=None):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        primitives = op_names if op_names else ALL_PRIMITIVES
        for primitive in primitives:
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
        assert x.shape[1] == self.C_in, f"Expected {self.C_in} input channels, got {x.shape[1]}"
        out = sum(w * op(x) for w, op in zip(weights, self._ops) if w > 1e-3)
        return self.proj(out)


class Cell(nn.Module):
    def __init__(self, C_prev, C_curr, reduction, cell_type='CNN'):
        super(Cell, self).__init__()
        self._ops = nn.ModuleList()
        self._indices = []
        cell_config = SEARCH_SPACE.get_cell_config(cell_type)
        self.n_nodes = cell_config.get('n_nodes', 4)
        self.n_inputs = cell_config.get('n_inputs', 2)
        op_names = SEARCH_SPACE.get_operations(cell_type)

        # Preprocess inputs to match C_curr
        self.preprocess = nn.ModuleList()
        for _ in range(self.n_inputs):
            self.preprocess.append(
                nn.Sequential(
                    nn.Conv2d(C_prev, C_curr, 1, bias=False),
                    nn.BatchNorm2d(C_curr)
                ) if C_prev != C_curr else nn.Identity()
            )

        # Operations for each edge
        for i in range(self.n_nodes):
            for j in range(self.n_inputs + i):
                stride = 2 if reduction and j < self.n_inputs else 1
                # All inputs are preprocessed to C_curr, or previous nodes output C_curr
                C_in = C_curr  # Always C_curr since inputs are adjusted
                op = MixedOp(C_in, C_curr, stride, op_names)
                self._ops.append(op)
                self._indices.append(j)

    def forward(self, inputs, weights):
        # Preprocess initial inputs to C_curr
        states = [self.preprocess[i](inputs[i]) for i in range(len(inputs))]
        offset = 0
        assert len(self._ops) == len(weights), f"Mismatch between ops ({len(self._ops)}) and weights ({len(weights)})"

        for i in range(self.n_nodes):
            curr_states = states[:self.n_inputs + i]
            curr_weights = weights[offset:offset + len(curr_states)]
            #print(f"Node {i}, Input states: {[s.shape for s in curr_states]}")  # Debug
            s = sum(self._ops[offset + j](h, curr_weights[j])
                    for j, h in enumerate(curr_states))
            #print(f"Node {i}, Output state: {s.shape}")  # Debug
            offset += len(curr_states)
            states.append(s)

        return torch.cat(states[-2:], dim=1)


class MicroDARTS(nn.Module):
    def __init__(self, num_classes=10):
        super(MicroDARTS, self).__init__()
        arch_config = SEARCH_SPACE.config['architecture']
        cnn_config = SEARCH_SPACE.get_cell_config('CNN')
        self.layers = arch_config['layers']  # 8
        channels = cnn_config['channels']
        C_curr = channels['initial']  # 8

        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
            nn.ReLU(inplace=True)
        )

        self.cells = nn.ModuleList()
        reduction_cells = cnn_config.get('reduction_cells', [])  # [2, 5]
        C_prev = C_curr  # 8 from stem

        for i in range(self.layers):
            reduction = i in reduction_cells
            cell = Cell(C_prev, C_curr, reduction, cell_type='CNN')
            self.cells.append(cell)
            num_ops = len(cell._ops)
            self.register_parameter(f'alpha_{i}', nn.Parameter(torch.randn(num_ops)))
            C_prev = C_curr * 2  # Concatenation doubles channels
            if reduction:
                C_curr *= channels['increment']  # 2

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        x = self.stem(x)  # [64, 8, 28, 28]
        #print(f"Stem output: {x.shape}")  # Debug
        for i, cell in enumerate(self.cells):
            weights = F.softmax(getattr(self, f'alpha_{i}'), dim=0)
            x = cell([x, x], weights)
            #print(f"After cell {i}: {x.shape}")  # Debug
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_mnist_loader(batch_size=64, data_root="data"):
    assert os.path.exists(os.path.join(data_root, "MNIST/raw")), "❌ MNIST data not found in 'data/MNIST/raw'"
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(root=data_root, train=True, download=False, transform=transform)
    test_set = torchvision.datasets.MNIST(root=data_root, train=False, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2,
                                              pin_memory=True)
    return train_loader, test_loader


def train(model, train_loader, optimizer, criterion):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def main():
    try:
        print("🚀 Running MicroDARTS on", device)
        training_config = SEARCH_SPACE.config['training']
        batch_size = training_config['batch_size']  # 64
        lr = training_config['learning_rate']  # 0.001

        train_loader, test_loader = get_mnist_loader(batch_size=batch_size)
        model = MicroDARTS().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=training_config['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        print("✅ Model Initialized. Starting Training...\n")

        epochs = 5  # For debugging
        for epoch in range(epochs):
            print(f"🔄 Epoch {epoch + 1}/{epochs}: Training...")
            train(model, train_loader, optimizer, criterion)
            acc = evaluate(model, test_loader)
            print(f"🎯 Epoch {epoch + 1}: Test Accuracy = {acc:.2f}%\n")

        best_architecture = [F.softmax(getattr(model, f'alpha_{i}'), dim=0).argmax().item() for i in
                             range(model.layers)]
        cnn_ops = SEARCH_SPACE.get_operations('CNN')
        print("\n🔥 Best Architecture Found:", [cnn_ops[idx] for idx in best_architecture])
        print("\n✅ Training Complete!")
    except Exception as e:
        logging.error("An error occurred", exc_info=True)


if __name__ == "__main__":
    main()