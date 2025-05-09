import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mc_darts import MicroDARTS, device
from normal_cnn import CNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from math import log2

# Define PIT-based Convolution Layer

class PITConv(nn.Module):
    def __init__(self, Cin, Cout, kernel_size, max_dilation):
        super().__init__()
        self.conv = nn.Conv2d(Cin, Cout, kernel_size, padding=kernel_size // 2, bias=False)
        self.alpha = nn.Parameter(torch.ones(Cout))  # Channel mask
        self.beta = nn.Parameter(torch.ones(kernel_size))  # Receptive field mask
        self.gamma = nn.Parameter(torch.ones(log2(max_dilation)))  # Dilation mask
        self.bn = nn.BatchNorm2d(Cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        QA = (self.alpha > 0.5).float().view(-1, 1, 1, 1)
        QB = (self.beta > 0.5).float().view(1, 1, -1, 1)
        QG = (self.gamma > 0.5).float().view(-1, 1, 1, 1)

        # Ensure all masks match the convolutional weights
        QA = QA.expand_as(self.conv.weight)
        QB = QB.expand_as(self.conv.weight)
        QG = QG.expand_as(self.conv.weight)

        masked_weights = self.conv.weight * QA * QB * QG
        out = F.conv2d(x, masked_weights, padding=self.conv.padding)

        return self.relu(self.bn(out))


# Convert DARTS model to PITDARTS
class PITDARTS(nn.Module):
    def __init__(self, init_channels=8, num_classes=10, layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(PITConv(1, init_channels, kernel_size=3, max_dilation=4))  # First layer with 1 input channel
        for _ in range(1, layers):
            self.layers.append(PITConv(init_channels, init_channels * 2, kernel_size=3, max_dilation=4))
            init_channels *= 2
        self.classifier = nn.Linear(init_channels, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return self.classifier(x)

# Load MNIST Data
def get_mnist_loader(batch_size=32, data_root="data/MNIST"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(root=data_root, train=True, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader

# Load Pretrained Model
def load_pretrained_model(saved_model_path, model_type="darts"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading pre-trained DARTS model...")
    if model_type == "darts":
        model = MicroDARTS(init_channels=8, num_classes=10, layers=4).to(device)
    else:
        model = CNN().to(device)
    checkpoint = torch.load(saved_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("✅ Pre-trained DARTS model loaded.")
    return model

# Transfer Weights
def transfer_weights(old_model, new_model):
    old_dict = old_model.state_dict()
    new_dict = new_model.state_dict()
    for key in new_dict.keys():
        if key in old_dict and old_dict[key].shape == new_dict[key].shape:
            new_dict[key] = old_dict[key]
    new_model.load_state_dict(new_dict)
    print("✅ Transferred pre-trained weights to PIT model.")

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

# Fine-Tune the PIT Model
def fine_tune_pit_model(pit_model, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pit_model.parameters(), lr=0.001)
    for epoch in range(5):  # Fine-tune for 5 epochs
        train_pit(pit_model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch + 1} completed.")
    return pit_model

# Save the Optimized Model
def save_pit_model(model, path="mnist_pit_optimized.pth"):
    print(f"Saving PIT-optimized model to '{path}'...")
    torch.save({'model_state': model.state_dict()}, path)
    print(f"✅ PIT-optimized model saved as '{path}'")

# Main Function
def main():
    saved_model_path = "trained-models/mnist_cnn.pth"
    train_loader = get_mnist_loader(batch_size=32, data_root="data")
    model_type = "cnn"
    model = load_pretrained_model(saved_model_path, model_type)
    
    print("Converting to PIT-enabled model...")
    pit_model = PITDARTS(init_channels=8, num_classes=10, layers=4).to(device)
    transfer_weights(model, pit_model)
    
    print("Fine-tuning the PIT model...")
    pit_model = fine_tune_pit_model(pit_model, train_loader)
    
    save_pit_model(pit_model, "trained-models/mnist_pit_normal_cnn.pth")

if __name__ == "__main__":
    main()
