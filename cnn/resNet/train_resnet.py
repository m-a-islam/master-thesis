import logging

import torch, os, sys
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torchvision import datasets, transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cnn.resNet.resnet_example import get_data_loaders
from cnn.resNet.mask_resnet import MaskedResNet
from cnn.resNet.utils import calculate_cost

#initialize logger
logging.basicConfig(filename="output/network_architecture.log", level=logging.INFO)
logger = logging.getLogger()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model and optimizer
    masked_resnet = MaskedResNet([2, 2, 2]).to(device)  # 3 layers
    optimizer = torch.optim.Adam(masked_resnet.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # CIFAR-10 data loader
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    train_loader, test_loader = get_data_loaders('./data', 64)

    # Example input tensor for CIFAR-10 (3x32x32)
    input_tensor = torch.randn(1, 3, 32, 32).to(device)

    epoch_list = []
    loss_list = []
    accuracy_list = []
    mask_values_list = []

    logger.info(f"Initial network before training:")
    logger.info(str(masked_resnet))

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

        logger.info(f"Final network after training:")
        logger.info(str(masked_resnet))
        # Log epoch data
        epoch_list.append(epoch + 1)
        loss_list.append(running_loss / len(train_loader))
        accuracy_list.append(100 * correct / total)
        mask_values_list.append(masked_resnet.mask.data.cpu().numpy())

        # Print mask values
        for i, mask in enumerate(masked_resnet.mask):
            if isinstance(mask, torch.Tensor):
                print(f"Mask {i} values (learnable tensor): {mask.data}")
            else:
                print(f"Mask {i} is not a tensor, value: {mask}")

        plot_logs(epoch_list, loss_list, accuracy_list, mask_values_list)

def plot_logs(epochs, loss, accuracy, mask_values):
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, accuracy, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()

    # Plot mask values
    plt.subplot(1, 3, 3)
    mask_values = np.array(mask_values)
    for i in range(mask_values.shape[1]):
        plt.plot(epochs, mask_values[:, i], label=f'Mask {i}')
    plt.xlabel('Epoch')
    plt.ylabel('Mask Value')
    plt.title('Mask Values')
    plt.legend()

    plt.tight_layout()
    plt.savefig('saved_model/masked_metrics_plot.png')
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
