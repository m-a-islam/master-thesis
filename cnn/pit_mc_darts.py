import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mc_darts import MicroDARTS  # Your existing DARTS model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define PIT-based Convolution Layer
class PITConv(nn.Module):
    def __init__(self, Cin, Cout, kernel_size, max_dilation):
        super().__init__()
        self.conv = nn.Conv2d(Cin, Cout, kernel_size, padding=kernel_size // 2, bias=False)
        self.alpha = nn.Parameter(torch.ones(Cout))  # Channel mask
        self.beta = nn.Parameter(torch.ones(kernel_size))  # Receptive field mask
        self.gamma = nn.Parameter(torch.ones(int(max_dilation)))  # Dilation mask
        self.bn = nn.BatchNorm2d(Cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        QA = (self.alpha > 0.5).float().view(-1, 1, 1, 1)
        QB = (self.beta > 0.5).float().view(1, 1, -1, 1)
        QG = (self.gamma > 0.5).float().view(1, 1, -1, 1)

        masked_weights = self.conv.weight * QA * QB * QG
        out = F.conv2d(x, masked_weights, padding=self.conv.padding)
        return self.relu(self.bn(out))

# Convert DARTS model to PITDARTS
class PITDARTS(nn.Module):
    def __init__(self, init_channels=8, num_classes=10, layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(PITConv(init_channels, init_channels * 2, kernel_size=3, max_dilation=4))
        self.classifier = nn.Linear(init_channels * 2, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return self.classifier(x)

# Load Pretrained Model
saved_model_path = "mnist_darts_checkpoint.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading pre-trained DARTS model...")
model = MicroDARTS(init_channels=8, num_classes=10, layers=4).to(device)
checkpoint = torch.load(saved_model_path, map_location=device)
model.load_state_dict(checkpoint['model_state'])
print("✅ Pre-trained DARTS model loaded.")

# Convert to PITDARTS model
print("Converting to PIT-enabled model...")
pit_model = PITDARTS(init_channels=8, num_classes=10, layers=4).to(device)

def transfer_weights(old_model, new_model):
    old_dict = old_model.state_dict()
    new_dict = new_model.state_dict()
    for key in new_dict.keys():
        if key in old_dict and old_dict[key].shape == new_dict[key].shape:
            new_dict[key] = old_dict[key]
    new_model.load_state_dict(new_dict)
    print("✅ Transferred pre-trained weights to PIT model.")

transfer_weights(model, pit_model)

# Define PIT Regularization
lambda_pit = 0.01
def PIT_regularizer(model):
    loss = 0
    for layer in model.layers:
        loss += torch.sum(torch.abs(layer.alpha))  # Channel pruning
        loss += torch.sum(torch.abs(layer.beta))   # Receptive field pruning
        loss += torch.sum(torch.abs(layer.gamma))  # Dilation pruning
    return loss

# Define Training Function
def train_pit(model, train_loader, criterion, optimizer):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels) + lambda_pit * PIT_regularizer(model)
        loss.backward()
        optimizer.step()

# Load MNIST Data
def get_mnist_loader(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader

# Fine-Tune the PIT Model
print("Fine-tuning the PIT model...")
train_loader = get_mnist_loader(batch_size=32)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pit_model.parameters(), lr=0.001)

for epoch in range(5):  # Fine-tune for 5 epochs
    train_pit(pit_model, train_loader, criterion, optimizer)
    print(f"Epoch {epoch + 1} completed.")

# Save the Optimized Model
torch.save({'model_state': pit_model.state_dict()}, "mnist_pit_optimized.pth")
print("✅ PIT-optimized model saved as 'mnist_pit_optimized.pth'")
