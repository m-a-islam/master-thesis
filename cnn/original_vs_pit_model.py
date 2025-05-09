import torch
from mc_darts import MicroDARTS, device, evaluate
from pit_mc_darts import PITDARTS, get_mnist_loader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from normal_cnn import CNN


# Load the original DARTS model
def load_original_model(saved_model_path, model_type="darts"):
    if model_type == "darts":
        model = MicroDARTS(init_channels=8, num_classes=10, layers=4).to(device)
    elif model_type == "cnn":
        model = CNN().to(device)
    else:
        raise ValueError("Invalid model type. Choose 'darts' or 'cnn'.")
    checkpoint = torch.load(saved_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model


# Load the PIT-optimized model
def load_pit_model(saved_model_path):
    model = PITDARTS(init_channels=8, num_classes=10, layers=4).to(device)
    checkpoint = torch.load(saved_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    return model


def get_mnist_loader(batch_size=32, data_root="data/MNIST"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(root=data_root, train=True, download=False, transform=transform)
    test_set = datasets.MNIST(root=data_root, train=False, download=False, transform=transform)

    # Split the training set into training and validation sets
    train_size = int(0.8 * len(train_set))
    valid_size = len(train_set) - train_size
    train_set, valid_set = random_split(train_set, [train_size, valid_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

# Main function to compare accuracies
def compare_accuracies():
    saved_original_model_path = "trained-models/mnist_cnn.pth"
    saved_pit_model_path = "trained-models/mnist_pit_normal_cnn.pth"

    # Load data
    train_loader, valid_loader, test_loader = get_mnist_loader(batch_size=32, data_root="data")

    # Evaluate original DARTS model
    model_type = "cnn"
    original_model = load_original_model(saved_original_model_path, model_type)
    print("Evaluating original DARTS model...")
    darts_valid_acc = evaluate(original_model, valid_loader, split_name="Valid")
    darts_test_acc = evaluate(original_model, test_loader, split_name="Test")

    # Evaluate PIT-optimized model
    pit_model = load_pit_model(saved_pit_model_path)
    print("Evaluating PIT-optimized model...")
    pit_valid_acc = evaluate(pit_model, valid_loader, split_name="Valid")
    pit_test_acc = evaluate(pit_model, test_loader, split_name="Test")

    # Compare accuracies
    print(f"\nOriginal DARTS Model - Valid Accuracy: {darts_valid_acc:.2f}%, Test Accuracy: {darts_test_acc:.2f}%")
    print(f"PIT-Optimized Model - Valid Accuracy: {pit_valid_acc:.2f}%, Test Accuracy: {pit_test_acc:.2f}%")

if __name__ == "__main__":
    compare_accuracies()