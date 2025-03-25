import torch, sys, os
import torch.nn as nn
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from mask_mobile_net import MobileNetV2
from cnn.resNet.utils import calculate_cost
from cnn.resNet.resnet_example import get_data_loaders

# Setup logging
logging.basicConfig(filename='output/mobilenetv2_architecture.log', level=logging.INFO)
logger = logging.getLogger()


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, optimizer and loss function
    masked_mobilenetv2 = MobileNetV2(num_classes=10).to(device)
    optimizer = torch.optim.Adam(masked_mobilenetv2.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # CIFAR-10 dataset loader
    train_loader, test_loader = get_data_loaders('./data', 64)

    # Example input tensor ensuring batch size > 1 for MAC calculation
    input_tensor = torch.randn(2, 3, 32, 32).to(device)

    # Log initial network structure
    logger.info("Initial Network (before training):")
    logger.info(str(masked_mobilenetv2))

    # Training loop (10 epochs)
    num_epochs = 10
    for epoch in range(num_epochs):
        masked_mobilenetv2.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = masked_mobilenetv2(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Compute statistics and print them
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {epoch_acc:.3f}%")

        # Calculate and log MACs and Model Size in eval mode to handle BatchNorm issue properly
        masked_mobilenetv2.eval()
        with torch.no_grad():
            macs, size = calculate_cost(masked_mobilenetv2, input_tensor)
        masked_mobilenetv2.train()

        print(f"Epoch {epoch + 1}: MACs: {macs}, Model Size: {size:.2f} MB")

    # Log the final architecture
    logger.info("Final Network (after training):")
    logger.info(str(masked_mobilenetv2))


if __name__ == "__main__":
    main()
