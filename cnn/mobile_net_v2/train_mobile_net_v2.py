import torch, sys, os
import torch.nn as nn
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from mask_mobile_net import MobileNetV2
from cnn.resNet.utils import calculate_cost
from cnn.resNet.resnet_example import get_data_loaders
from thop import profile
from torch import optim

# Setup logging
logging.basicConfig(filename='output/mobilenetv2_architecture.log', level=logging.INFO)
logger = logging.getLogger()

def calculate_block_contributions(model, input_size=(1, 3, 32, 32)):
    """Precompute MACs and parameter counts for each block."""
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    macs_blocks = []
    params_blocks = []
    blocks = [model.block1, model.block2, model.block3, model.block4, 
              model.block5, model.block6, model.block7]
    
    # Temporarily set all masks to 1 to compute full contributions
    original_masks = model.mask.data.clone()
    model.mask.data.fill_(10.0)  # sigmoid(10) ≈ 1
    
    for i, block in enumerate(blocks):
        # Compute MACs for this block
        macs, _ = profile(block, inputs=(dummy_input,), verbose=False)
        macs_blocks.append(macs)
        # Compute parameters for this block
        params = sum(p.numel() for p in block.parameters())
        params_blocks.append(params)
        # Update input for the next block
        dummy_input = block(dummy_input)
    
    # Restore original masks
    model.mask.data.copy_(original_masks)
    
    # Compute fixed parts (conv1, conv2, etc.)
    fixed_parts = nn.Sequential(model.conv1, model.bn1, model.relu, 
                                model.conv2, model.bn2, model.avgpool, model.fc)
    macs_fixed, _ = profile(fixed_parts, inputs=(torch.randn(input_size).to(device),), verbose=False)
    params_fixed = sum(p.numel() for p in fixed_parts.parameters())
    
    return macs_blocks, params_blocks, macs_fixed, params_fixed

def train_with_constraints(model, train_loader, criterion, optimizer, device, 
                          lambda_macs, lambda_size, macs_blocks, params_blocks, 
                          macs_fixed, params_fixed):
    """Train the model with penalties to enforce MACs and size constraints."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        classification_loss = criterion(outputs, labels)

        # Compute expected MACs and size
        mask_weights = torch.sigmoid(model.mask)
        expected_macs = macs_fixed + sum(mask_weights[i] * macs_blocks[i] for i in range(7))
        expected_size = (params_fixed + sum(mask_weights[i] * params_blocks[i] for i in range(7))) * 4 / 1e6  # MB

        # Penalties for exceeding thresholds
        macs_penalty = torch.relu(expected_macs - model.mac_threshold)
        size_penalty = torch.relu(expected_size - model.size_threshold)

        # Total loss
        loss = classification_loss + lambda_macs * macs_penalty + lambda_size * size_penalty
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with thresholds
    size_threshold = 5.0  # MB
    mac_threshold = 5e6   # Example MACs threshold (adjust as needed)
    model = MobileNetV2(num_classes=10, size_threshold=size_threshold, 
                        mac_threshold=mac_threshold).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Dummy train_loader (replace with actual data loader)
    from torch.utils.data import DataLoader, TensorDataset
    dummy_data = torch.randn(1000, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (1000,))
    train_loader = DataLoader(TensorDataset(dummy_data, dummy_labels), batch_size=32, shuffle=True)

    # Precompute MACs and parameter contributions
    macs_blocks, params_blocks, macs_fixed, params_fixed = calculate_block_contributions(model)
    
    # Hyperparameters for penalties
    lambda_macs = 1e-6  # Adjust based on scale of MACs
    lambda_size = 1e-3  # Adjust based on scale of size

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        loss, acc = train_with_constraints(model, train_loader, criterion, optimizer, device,
                                          lambda_macs, lambda_size, macs_blocks, params_blocks,
                                          macs_fixed, params_fixed)
        
        # Compute current MACs and size for logging
        mask_weights = torch.sigmoid(model.mask)
        current_macs = macs_fixed + sum(mask_weights[i] * macs_blocks[i] for i in range(7))
        current_size = (params_fixed + sum(mask_weights[i] * params_blocks[i] for i in range(7))) * 4 / 1e6
        
        logging.info(f"Epoch {epoch+1}: Loss: {loss:.4f}, Accuracy: {acc:.2f}%, "
                     f"MACs: {current_macs:.2e}, Size: {current_size:.2f} MB")
        logging.info(f"Network: {model.get_network_description()}")
        
        # Check if thresholds are met
        if current_macs <= mac_threshold and current_size <= size_threshold:
            logging.info("Thresholds satisfied!")
            break


if __name__ == "__main__":
    main()
