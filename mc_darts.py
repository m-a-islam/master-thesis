import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision, os
import torchvision.transforms as transforms

# Set device for CUDA (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
## todo change the search space for fusion and mlp, cnn separately
# Define the search space
PRIMITIVES = [
    'none',                      # No connection
    'skip_connect',              # Identity (ResNet-style skip)
    'max_pool_2x2',              # Max Pooling (Downsampling)
    'avg_pool_2x2',              # Average Pooling
    'conv_1x1',                  # Pointwise Convolution
    'conv_3x3',                  # Standard 3x3 Convolution
    'depthwise_conv_3x3',        # Depthwise Separable Conv
    'dilated_conv_3x3',          # Dilated Convolution
    'grouped_conv_3x3',          # Grouped Convolution
    'conv_5x5',                  # Larger 5x5 Convolution
    'conv_7x7',                  # Very Large 7x7 Convolution
    'mlp',                       # Fully Connected MLP
    'squeeze_excitation',        # SE Layer for Attention
    'batch_norm',                # Batch Normalization Only
    'layer_norm',                # Layer Normalization
    'dropout'                    # Dropout Layer for Regularization
]

## todo change the Opearation according to the seach space
# Define Operations Dictionary
OPS = {
    'none': lambda C, stride: nn.Identity(),
    'skip_connect': lambda C, stride: nn.Identity() if stride == 1 else nn.Conv2d(C, C, 1, stride=stride),
    'max_pool_2x2': lambda C, stride: nn.MaxPool2d(2, stride=stride),
    'avg_pool_2x2': lambda C, stride: nn.AvgPool2d(2, stride=stride),
    'conv_1x1': lambda C, stride: nn.Conv2d(C, C, 1, stride=stride, padding=0, bias=False),
    'conv_3x3': lambda C, stride: nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False),
    'depthwise_conv_3x3': lambda C, stride: nn.Conv2d(C, C, 3, stride=stride, padding=1, groups=C, bias=False),
    'dilated_conv_3x3': lambda C, stride: nn.Conv2d(C, C, 3, stride=stride, padding=2, dilation=2, bias=False),
    'grouped_conv_3x3': lambda C, stride: nn.Conv2d(C, C, 3, stride=stride, padding=1, groups=2, bias=False),
    'conv_5x5': lambda C, stride: nn.Conv2d(C, C, 5, stride=stride, padding=2, bias=False),
    'conv_7x7': lambda C, stride: nn.Conv2d(C, C, 7, stride=stride, padding=3, bias=False),
    'mlp': lambda C, stride: nn.Sequential(
        nn.Flatten(),
        nn.Linear(C * 4 * 4, C),
        nn.ReLU(),
        nn.Linear(C, C)
    ),
    'squeeze_excitation': lambda C, stride: nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(C, C // 16, 1),
        nn.ReLU(),
        nn.Conv2d(C // 16, C, 1),
        nn.Sigmoid()
    ),
    'batch_norm': lambda C, stride: nn.BatchNorm2d(C),
    'layer_norm': lambda C, stride: nn.LayerNorm([C, 28, 28]),
    'dropout': lambda C, stride: nn.Dropout(p=0.3)
}

class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()

        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride).to(device)  # Move operations to GPU
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Ensure weights is correctly applied to operations.
        """
        weights = weights.view(-1)  # Ensure weights is a 1D vector
        return sum(w * op(x) for w, op in zip(weights, self._ops))

## todo update cell for multiple types of search space
class Cell(nn.Module):
    def __init__(self, C, reduction):
        super(Cell, self).__init__()
        self._ops = nn.ModuleList()
        self._indices = []

        for i in range(6):  # Increase nodes per cell
            for j in range(3 + i):  # More connections per node
                op = MixedOp(C, stride=2 if reduction else 1).to(device)
                self._ops.append(op)
                self._indices.append(j)

    def forward(self, inputs, weights):
        states = [inputs[0], inputs[1]]

        new_states = []
        op_index = 0
        for i in range(6):  # Increase number of processed states
            sum_result = 0
            for j in range(3 + i):
                if j < len(states):
                    sum_result += self._ops[op_index](states[j], weights[op_index])
                op_index += 1
            new_states.append(sum_result)

        states.extend(new_states)
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