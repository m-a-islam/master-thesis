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
import logging

# Configure logging
logging.basicConfig(filename='error-file.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(message)s')

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
        
        # Get cell configuration
        cell_config = SEARCH_SPACE.get_cell_config(cell_type)
        self.n_nodes = cell_config.get('n_nodes', 4)
        self.n_inputs = cell_config.get('n_inputs', 2)
        
        # Get operations specific to this cell type
        op_names = SEARCH_SPACE.get_operations(cell_type)
        
        # Create mixed operations
        for i in range(self.n_nodes):
            for j in range(self.n_inputs + i):
                stride = 2 if reduction and j < self.n_inputs else 1
                # Ensure proper channel handling in MixedOp
                op = MixedOp(C, stride, op_names)
                self._ops.append(op)
                self._indices.append(j)

    def forward(self, inputs, weights):
        states = list(inputs)  # Convert tuple to list if needed
        offset = 0
        
        # Ensure weights match the number of operations
        assert len(self._ops) == len(weights), f"Mismatch between ops ({len(self._ops)}) and weights ({len(weights)})"
        
        for i in range(self.n_nodes):
            curr_states = states[:self.n_inputs + i]
            curr_weights = weights[offset:offset + len(curr_states)]
            
            # Sum the weighted operations
            s = sum(self._ops[offset + j](h, curr_weights[j])
                   for j, h in enumerate(curr_states))
            
            offset += len(curr_states)
            states.append(s)
        
        return torch.cat(states[-2:], dim=1)

class MicroDARTS(nn.Module):
    def __init__(self, C=16, num_classes=10, layers=8):
        super(MicroDARTS, self).__init__()
        self.layers = layers
        
        # Read configuration from search space
        config = SEARCH_SPACE.get_cell_config('CNN')
        channels = config['channels']
        C_curr = channels['initial']  # Starting with 16 channels
        
        # Initial stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        # Create cells with proper channel handling
        self.cells = nn.ModuleList()
        reduction_cells = config.get('reduction_cells', [])
        C_prev = C_curr
        
        for i in range(layers):
            reduction = i in reduction_cells
            if reduction:
                C_curr *= channels['increment']  # Double channels after reduction
                
            cell = Cell(C_prev, reduction, cell_type='CNN')
            self.cells.append(cell)
            
            # Initialize alphas for each cell
            num_ops = len(cell._ops)
            self.register_parameter(f'alpha_{i}', nn.Parameter(torch.randn(num_ops)))
            
            C_prev = C_curr  # Update channel count for next cell

        # Global pooling and classifier
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_curr, num_classes)

    def forward(self, x):
        x = self.stem(x)
        
        for i, cell in enumerate(self.cells):
            weights = F.softmax(getattr(self, f'alpha_{i}'), dim=0)
            x = cell([x, x], weights)

        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
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
    try:
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
    except Exception as e:
        logging.error("An error occurred", exc_info=True)

if __name__ == "__main__":
    main()