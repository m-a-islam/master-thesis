import torch, sys, os
import torch.nn as nn
import logging, torch.onnx
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from mask_mobile_net import MobileNetV2
from cnn.resNet.utils import calculate_cost
from cnn.resNet.resnet_example import get_data_loaders
from thop import profile
from torch import optim

# Setup logging
logging.basicConfig(filename='output/mobilenetv2_architecture.log', level=logging.INFO)
logger = logging.getLogger()

def calculate_block_contributions(model, input_size=(2, 3, 32, 32)):
    # Get the device of the model
    device = next(model.parameters()).device
    # Create a dummy input tensor
    dummy_input = torch.randn(input_size).to(device)
    
    # Profile initial layers (conv1, bn1, relu)
    initial_layers = nn.Sequential(model.conv1, model.bn1, model.relu)
    macs_initial, _ = profile(initial_layers, inputs=(dummy_input,), verbose=False)
    params_initial = sum(p.numel() for p in initial_layers.parameters())
    
    # Pass input through initial layers
    x = initial_layers(dummy_input)
    
    # Define and profile each block sequentially
    blocks = [model.block1, model.block2, model.block3, model.block4, 
              model.block5, model.block6, model.block7]
    macs_blocks = []
    params_blocks = []
    
    for block in blocks:
        macs, _ = profile(block, inputs=(x,), verbose=False)
        macs_blocks.append(macs)
        params = sum(p.numel() for p in block.parameters())
        params_blocks.append(params)
        x = block(x)  # Update the input for the next block
    
    # Profile final layers (conv2, bn2, avgpool, fc)
    final_layers = nn.Sequential(model.conv2, model.bn2, model.avgpool, nn.Flatten(start_dim=1), model.fc)
    macs_final, _ = profile(final_layers, inputs=(x,), verbose=False)
    params_final = sum(p.numel() for p in final_layers.parameters())
    
    # Combine initial and final contributions into fixed parts
    macs_fixed = macs_initial + macs_final
    params_fixed = params_initial + params_final
    print(f"macs_blocks: {macs_blocks}")
    print(f"params_blocks: {params_blocks}")
    print(f"macs_fixed: {macs_fixed}")
    print(f"params_fixed: {params_fixed}")
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

        macs_penalty = macs_penalty.to(device)
        size_penalty = size_penalty.to(device)


        #print(f"classification_loss device: {classification_loss.device}")

        # Total loss
        loss = classification_loss + lambda_macs * macs_penalty + lambda_size * size_penalty
        #print(f"Before backward: mask = {model.mask.data}")
        loss.backward()
        #print(f"After backward: mask = {model.mask.data}")

        optimizer.step()
        #print(f"After optimizer step: mask = {model.mask.data}")
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
    size_threshold = 2.0  # MB
    mac_threshold = 10000   # Example MACs threshold (adjust as needed)
    # todo: accuracy threshold should be implemented here
    model = MobileNetV2(num_classes=10, size_threshold=size_threshold, mac_threshold=mac_threshold).to(device)


    #torch.save(model.state_dict(), f"output/before_train_mobile_net_v2.pth")
    #convert_to_onnx(model, f"output/onnx/before_train_mobile_net_v2.onnx")

    # for name, param in model.named_parameters():
    #     print(f"name: {name}: {param.device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)

    # for group in optimizer.param_groups:
    #     for p in group['params']:
    #         state = optimizer.state[p]
    #         if state:
    #             print(f"Parameter {p.device}, exp_avg: {state['exp_avg'].device}")



    train_loader, _ = get_data_loaders('./data')

    # Precompute MACs and parameter contributions
    macs_blocks, params_blocks, macs_fixed, params_fixed = calculate_block_contributions(model)
    
    # Hyperparameters for penalties
    lambda_macs = 1e-4  # Adjust based on scale of MACs
    lambda_size = 1e-2  # Adjust based on scale of size

    # Log initial architecture
    print("Before Training Architecture:", model.get_network_description())
    initial_macs = macs_fixed + sum(torch.sigmoid(model.mask[i]) * macs_blocks[i] for i in range(7))
    initial_size = (params_fixed + sum(torch.sigmoid(model.mask[i]) * params_blocks[i] for i in range(7))) * 4 / 1e6
    print(f"Initial MACs: {initial_macs:.2e}, Initial Size: {initial_size:.2f} MB")

    # Metric tracking lists
    losses = []
    accuracies = []
    macs_list = []
    sizes_list = []

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        loss, acc = train_with_constraints(model, train_loader, criterion, optimizer, device, lambda_macs, lambda_size, macs_blocks, params_blocks, macs_fixed, params_fixed)
        
        # Compute current MACs and size for logging
        mask_weights = torch.sigmoid(model.mask)
        current_macs = macs_fixed + sum(mask_weights[i] * macs_blocks[i] for i in range(7))
        current_size = (params_fixed + sum(mask_weights[i] * params_blocks[i] for i in range(7))) * 4 / 1e6

        # Store metrics
        # Append metrics, ensuring they are CPU scalars
        losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
        accuracies.append(acc.item() if isinstance(acc, torch.Tensor) else acc)
        macs_list.append(current_macs.cpu().item() if isinstance(current_macs, torch.Tensor) else current_macs)
        sizes_list.append(current_size.cpu().item() if isinstance(current_size, torch.Tensor) else current_size)

        logging.info(f"Epoch {epoch+1}: Loss: {loss:.4f}, Accuracy: {acc:.2f}%, "
                     f"MACs: {current_macs:.2e}, Size: {current_size:.2f} MB")
        logging.info(f"Network: {model.get_network_description()}")

        # todo: implement stopping criteria here
        # todo: currently stopped imidiately after the condition is met without any further training
        # Check if thresholds are met
        if current_macs <= mac_threshold and current_size <= size_threshold:
            logging.info("Thresholds satisfied!")
            continue


    torch.save(model.state_dict(), f"output/structural_prune_after_train_mobile_net_v2.pth")
    #convert_to_onnx(model, f"output/onnx/after_train_mobile_net_v2.onnx")
    # Log final architecture
    final_mask_weights = torch.sigmoid(model.mask)
    final_macs = macs_fixed + sum(final_mask_weights[i] * macs_blocks[i] for i in range(7))
    final_size = (params_fixed + sum(final_mask_weights[i] * params_blocks[i] for i in range(7))) * 4 / 1e6
    print(f"Final Architecture: {model.get_network_description()}")
    print(f"Final MACs: {final_macs:.2e}, Final Size: {final_size:.2f} MB")
    # Plot the metrics
    plot_metrics(len(losses), losses, accuracies, macs_list, sizes_list)
    """
    example_input = torch.randn(1, 3, 32, 32).to(device)
    onnx_path = f"output/onnx/structural_prune_after_train_mobile_net_v2.onnx"

    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ Model saved as ONNX to {onnx_path}")
    model.eval()
    with torch.no_grad():
        output = model(example_input)
        print(output.shape)
    """


# Plotting function
def plot_metrics(epochs, losses, accuracies, macs_list, sizes_list):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(range(1, epochs + 1), losses, label='Loss')
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    axs[0, 1].plot(range(1, epochs + 1), accuracies, label='Accuracy', color='orange')
    axs[0, 1].set_title('Training Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy (%)')
    axs[0, 1].legend()

    axs[1, 0].plot(range(1, epochs + 1), macs_list, label='MACs', color='green')
    axs[1, 0].set_title('Model MACs')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('MACs')
    axs[1, 0].legend()

    axs[1, 1].plot(range(1, epochs + 1), sizes_list, label='Size', color='red')
    axs[1, 1].set_title('Model Size (MB)')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Size (MB)')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig('output/mobile_net_soft_pruning_metrices.png')
    plt.show()


def convert_to_onnx(model, output_path):
    #dummy_input = torch.randn(1, 32, )  # cifar10 input shape
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=True
    )
    print(f"✅ Model saved as ONNX to {output_path}")



def load_model_and_convert_to_onnx(pth_path, onnx_path, input_size=(1, 3, 32, 32)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model architecture
    model = MobileNetV2(num_classes=10).to(device)

    # Load the saved state dictionary
    state_dict = torch.load(pth_path, map_location=device)
    #model.load_state_dict(state_dict)


    # Filter out keys with "total_ops" and "total_params"
    filtered_state_dict = {k: v for k, v in state_dict.items() if "total_ops" not in k and "total_params" not in k}

    # Check for size mismatches and adjust the model if necessary
    model_state_dict = model.state_dict()
    for key in filtered_state_dict.keys():
        if key in model_state_dict and filtered_state_dict[key].shape != model_state_dict[key].shape:
            print(
                f"Size mismatch for {key}: expected {model_state_dict[key].shape}, got {filtered_state_dict[key].shape}")
            return

    model.load_state_dict(filtered_state_dict)

    # Set the model to evaluation mode
    #model.to(device)
    model.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(input_size).to(device)

    # Convert the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ Model saved as ONNX to {onnx_path}")

# Example usage

if __name__ == "__main__":
    main()
    #load_model_and_convert_to_onnx('output/structural_prune_after_train_mobile_net_v2.pth', 'output/onnx/structural_prune_after_train_mobile_net_v2.onnx')