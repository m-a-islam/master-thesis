import torch
from ptflops import get_model_complexity_info

def test(model, test_loader, criterion, args):
    model.eval()
    macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
    print(f"Model MACs: {macs}, Parameters: {params}")
    
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = 100 * total_correct / total_samples
    print(f'Test Accuracy: {accuracy:.2f}%')