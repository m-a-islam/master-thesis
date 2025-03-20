import torch, sys, os
import torch.nn as nn
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from mask_mobile_net import MobileNetV2
from cnn.resNet.utils import calculate_cost
from cnn.resNet.resnet_example import get_data_loaders

# Initialize logger
logging.basicConfig(filename='output/mobilenetv2_architecture.log', level=logging.INFO)
logger = logging.getLogger()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model and optimizer
    masked_mobilenetv2 = MobileNetV2(num_classes=10).to(device)
    optimizer = torch.optim.Adam(masked_mobilenetv2.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # CIFAR-10 data loader
    train_loader, test_loader = get_data_loaders('./data', 64)

    # Example input tensor for CIFAR-10 (3x32x32)
    input_tensor = torch.randn(1, 3, 32, 32).to(device)

    # Log the initial network architecture
    logger.info("Initial Network (before any training):")
    logger.info(str(masked_mobilenetv2))

    # Training loop (10 epochs)
    for epoch in range(10):
        masked_mobilenetv2.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass through the masked model
            outputs = masked_mobilenetv2(inputs)
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
        macs, size = calculate_cost(masked_mobilenetv2, input_tensor)
        print(f"Epoch {epoch+1}: MACs: {macs}, Model Size: {size:.2f} MB")

    # After training, log the final network architecture
    logger.info("Final Network (after training and applying mask):")
    logger.info(str(masked_mobilenetv2))


# Run the main function
if __name__ == "__main__":
    main()
