import torch, sys, json
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.models import resnet18
from fvcore.nn import FlopCountAnalysis, parameter_count

matplotlib.use('Agg')

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loaders(data_dir, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=transform
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_model(model, train_loader, test_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_accuracies = []

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

        train_losses.append(running_loss / len(train_loader))
        test_accuracy = evaluate_model(model, test_loader, device, return_accuracy=True)
        test_accuracies.append(test_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {test_accuracy:.2f}%")

    # Save metrics to a file
    with open('saved_model/metrics.json', 'w') as f:
        json.dump({'train_losses': train_losses, 'test_accuracies': test_accuracies}, f)

    return model, train_losses, test_accuracies

def evaluate_model(model, test_loader, device, return_accuracy=False):
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
    if return_accuracy:
        return accuracy
    print(f'Model Accuracy: {accuracy:.2f}%')

def main():
    data_dir = './data'
    device = get_device()
    train_loader, test_loader = get_data_loaders(data_dir)

    model = resnet18(weights=None, num_classes=10).to(device)

    trained_model, train_losses, test_accuracies = train_model(model, train_loader, test_loader, device, epochs=10)
    torch.save(model.state_dict(), 'saved_model/resnet18_cifar10.pth')
    evaluate_model(trained_model, test_loader, device)

    plot_saved_metrics()

def load_model():
    model = resnet18(num_classes=10)
    model.load_state_dict(torch.load('saved_model/resnet18_cifar10.pth'))
    device = get_device()
    model.to(device)
    model.eval()
    return model


def check_flops_mac():
    model = load_model()
    # Create a dummy input tensor with the same shape as your input data
    dummy_input = torch.randn(1, 3, 32, 32)

    # Calculate FLOPs and MACs
    flops = FlopCountAnalysis(model, dummy_input)
    params = parameter_count(model)

    print(f"FLOPs: {flops.total()}")
    print(f"MACs: {flops.total()}")
    print(f"Parameters: {params['']}")

def visualize_model():
    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_loader, test_loader = get_data_loaders('./data')
    # evaluate_model(model, test_loader, device)
    model.to(device)
    summary(model, (3, 32, 32))
    # Redirect stdout to a file
    with open('saved_model/model_summary.txt', 'w') as f:
        sys.stdout = f
        summary(model, (3, 32, 32))
        sys.stdout = sys.__stdout__  # Reset redirect.


def plot_saved_metrics():
    # Load metrics from the file
    with open('saved_model/metrics.json', 'r') as f:
        metrics = json.load(f)

    train_losses = metrics['train_losses']
    test_accuracies = metrics['test_accuracies']

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, 'r', label='Test accuracy')
    plt.title('Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('saved_model/metrics_plot.png')
    plt.show()

if __name__ == "__main__":
    #main()
    #check_flops_mac()
    #visualize_model()

    # model = load_model()
    # data_dir = './data'
    # device = get_device()
    # train_loader, test_loader = get_data_loaders(data_dir)
    #
    # trained_model, train_losses, test_accuracies = train_model(model, train_loader, test_loader, device, epochs=10)
    plot_saved_metrics()