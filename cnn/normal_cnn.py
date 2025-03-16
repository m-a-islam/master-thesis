import torch
import torch.nn as nn
import torch.onnx as onnx
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.onnx as onnx
import os

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    os.makedirs("trained-models", exist_ok=True)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
    model_path = "trained-models/mnist_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Model Accuracy: {accuracy:.2f}%')

def convert_to_onnx(model, device):
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    os.makedirs("trained-models/onnx", exist_ok=True)
    onnx_filename = "trained-models/onnx/mnist_cnn.onnx"
    onnx.export(model, dummy_input, onnx_filename, export_params=True, opset_version=11, do_constant_folding=True,
                input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"Model converted to ONNX format and saved as {onnx_filename}")

def main():
    device = get_device()
    train_loader, test_loader = get_data_loaders()
    model = CNN().to(device)
    train_model(model, train_loader, device)
    evaluate_model(model, test_loader, device)
    convert_to_onnx(model, device)

if __name__ == "__main__":
    main()
