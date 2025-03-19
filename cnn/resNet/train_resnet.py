import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn.resNet.utils import masked_resnet


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model and optimizer
    masked_resnet = MaskedResNet([2, 2, 2]).to(device)  # 3 layers
    optimizer = torch.optim.Adam(masked_resnet.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # CIFAR-10 data loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    # Example input tensor for CIFAR-10 (3x32x32)
    input_tensor = torch.randn(1, 3, 32, 32).to(device)

    # Training loop (10 epochs)
    for epoch in range(10):
        masked_resnet.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass through the masked model
            outputs = masked_resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print stats
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {100 * correct / total}%")

        # Calculate and print costs (MACs and Size)
        macs, size = calculate_cost(masked_resnet, input_tensor)
        print(f"Epoch {epoch+1}: MACs: {macs}, Model Size: {size:.2f} MB")
        print(f"Mask values: {masked_resnet.mask.data}")

# Run the main function
if __name__ == "__main__":
    main()
