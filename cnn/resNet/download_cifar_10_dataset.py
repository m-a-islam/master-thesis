import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def main():
    # Define the directory to save the dataset
    data_dir = './data'

    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    # Download training dataset
    print("Downloading CIFAR-10 training dataset...")
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Download testing dataset
    print("Downloading CIFAR-10 testing dataset...")
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Verify dataset size and details
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    # Optional: create DataLoaders to confirm batch loading
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    print("First batch of training data loaded:")
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")  # Should be [64, 3, 32, 32]
        print(f"Label batch shape: {labels.shape}")  # Should be [64]
        break  # Only display the first batch


if __name__ == "__main__":
    main()
