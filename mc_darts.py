import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import os
import torchvision.transforms as transforms
from primitives import ALL_PRIMITIVES, PRIMITIVES
from operations import OPS
from phylum import SEARCH_SPACE

# Set device for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

class MixedOp(nn.Module):
    def __init__(self, C, stride, op_names=None):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        
        # Use provided op_names or default to ALL_PRIMITIVES
        primitives = op_names if op_names else ALL_PRIMITIVES
        
        for primitive in primitives:
            op = OPS[primitive](C, stride).to(device)
            self._ops.append(op)

    def forward(self, x, weights):
        weights = weights.view(-1)
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):
    def __init__(self, C, reduction, cell_type='CNN'):
        super(Cell, self).__init__()
        self._ops = nn.ModuleList()
        self._indices = []

        # Get cell configuration from search space
        cell_config = SEARCH_SPACE.get_cell_config(cell_type)
        n_nodes = cell_config.get('n_nodes', 6)
        n_inputs = cell_config.get('n_inputs', 2)

        # Get operations specific to this cell type
        op_names = SEARCH_SPACE.get_operations(cell_type)

        for i in range(n_nodes):
            for j in range(n_inputs + i):
                stride = 2 if reduction and j < n_inputs else 1
                op = MixedOp(C, stride, op_names).to(device)
                self._ops.append(op)
                self._indices.append(j)

    def forward(self, inputs, weights):
        states = [inputs[0], inputs[1]]
        offset = 0
        print(f"Debug: Number of operations: {len(self._ops)}")
        print(f"Debug: Weights shape: {weights.shape}")
        for i in range(len(self._indices) // len(states)):
            s = sum(self._ops[offset + j](h, weights[offset + j])
                   for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-2:], dim=1)

class MicroDARTS(nn.Module):
    def __init__(self, C=8, num_classes=10, layers=3):
        super(MicroDARTS, self).__init__()
        self.layers = layers
        self.stem = nn.Conv2d(1, C, 3, stride=1, padding=1, bias=False)

        self.cells = nn.ModuleList()
        for i in range(layers):
            reduction = (i % 2 == 1)
            self.cells.append(Cell(C, reduction))
            if reduction:
                C *= 2  # Double the number of channels after each reduction

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes).to(device)  # Adjusted to match the actual input size

        self.alpha_ops = nn.ParameterList([
            nn.Parameter(torch.randn(len(self.cells[i]._ops))) for i in range(layers)
        ])

    def forward(self, x):
        x = self.stem(x)

        for i, cell in enumerate(self.cells):
            weights = F.softmax(self.alpha_ops[i], dim=0)
            x = cell([x, x], weights)

        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)  # Flatten

        #print(f"🔍 Debug: Final feature vector shape before classifier: {x.shape}")  # Debugging

        return self.classifier(x)

# Optimized Data Loader for GPU
def get_mnist_loader(batch_size=64, data_root="data"):
    """
    Loads MNIST dataset from pre-downloaded folder data/MNIST/raw.
    - Assumes data/MNIST/raw contains train-images-idx3-ubyte, t10k-images-idx3-ubyte, etc.
    - Does NOT download data again.
    """
    # Ensure the data directory exists
    assert os.path.exists(os.path.join(data_root, "MNIST/raw")), "❌ MNIST data not found in 'data/MNIST/raw'"

    transform = transforms.Compose([
        transforms.RandomRotation(10),  # Random rotation
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = torchvision.datasets.MNIST(root=data_root, train=True, download=False, transform=transform)
    test_set = torchvision.datasets.MNIST(root=data_root, train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader

# Training function (now using CUDA)
def train(model, train_loader, optimizer, criterion):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move to GPU
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluation function (now using CUDA)
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move to GPU
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Main function (now using CUDA)
def main():
    print("🚀 Running MicroDARTS on", device)
    print("📚 Using Search Space Configuration:")
    print(f"Available CNN operations: {SEARCH_SPACE.get_operations('CNN')}")
    print(f"Available MLP operations: {SEARCH_SPACE.get_operations('MLP')}")
    print(f"Available Fusion operations: {SEARCH_SPACE.get_operations('Fusion')}")
    
    train_loader, test_loader = get_mnist_loader()

    # Initialize Model
    model = MicroDARTS().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("✅ Model Initialized. Starting Training...\n")

    for epoch in range(5):
        print(f"🔄 Epoch {epoch + 1}: Training...")
        train(model, train_loader, optimizer, criterion)
        acc = evaluate(model, test_loader)
        print(f"🎯 Epoch {epoch + 1}: Test Accuracy = {acc:.2f}%\n")

    best_architecture = [F.softmax(alpha, dim=0).argmax().item() for alpha in model.alpha_ops]
    best_architecture = [idx if idx < len(PRIMITIVES) else len(PRIMITIVES) - 1 for idx in best_architecture]

    print("\n🔥 Best Architecture Found:", [PRIMITIVES[idx] for idx in best_architecture])
    print("\n✅ Training Complete!")

if __name__ == "__main__":
    main()