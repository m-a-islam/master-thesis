from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 data loader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Basic training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
masked_resnet.to(device)
optimizer = torch.optim.Adam(masked_resnet.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):  # Example: 10 epochs
    masked_resnet.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = masked_resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {100 * correct / total}%")
