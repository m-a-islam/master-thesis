import torch
import torch.nn as nn
import torch.optim as optim
import os
from cnn.plinio.plinio.methods import SuperNet, PIT, MPS
from cnn.plinio.plinio.regularizers import BaseRegularizer

# Load Custom CNN Model from normal_cnn.py
from cnn.normal_cnn import CNN, get_device, get_data_loaders

# Define Optimization Type: Choose one -> 'supernet', 'pit', 'mps'
OPTIMIZATION_METHOD = 'pit'  # Change to 'supernet' or 'mps' if needed

# Define Constraint Type: Choose one -> 'latency', 'memory'
COST_MODEL = 'memory'  # Adjust based on your requirements

def optimize_model(model):
    """
    Wrap the given model with PLiNIO optimization based on the selected method.
    """
    if OPTIMIZATION_METHOD == 'supernet':
        return SuperNet(model, cost=COST_MODEL)
    elif OPTIMIZATION_METHOD == 'pit':
        return PIT(model, input_shape=(1, 28, 28), cost=COST_MODEL)
    elif OPTIMIZATION_METHOD == 'mps':
        return MPS(model, cost=COST_MODEL)
    else:
        raise ValueError("Invalid optimization method. Choose from ['supernet', 'pit', 'mps'].")

def train_optimized_model(model, train_loader, device, epochs=5):
    """
    Train the optimized model with PLiNIO's cost-aware optimization.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Apply Regularization (cost constraint)
    regularizer = BaseRegularizer(cost_name=COST_MODEL, strength=0.1)  

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            task_loss = criterion(outputs, labels)

            # Compute cost and apply constraint
            cost = model.cost()
            loss = regularizer(task_loss, cost)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    return model

def evaluate_model(model, test_loader, device):
    """
    Evaluate model accuracy on test dataset.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Optimized Model Accuracy: {accuracy:.2f}%')

def export_optimized_model(model):
    """
    Save the optimized model for deployment.
    """
    optimized_model = model.extract_model()
    os.makedirs("trained-models", exist_ok=True)
    model_path = "trained-models/optimized_cnn.pth"
    torch.save(optimized_model.state_dict(), model_path)
    print(f"Optimized model saved as {model_path}")

def main():
    """
    Main execution function for training and optimizing the CNN.
    """
    device = get_device()
    train_loader, test_loader = get_data_loaders()

    # Load original model and optimize it
    model = CNN().to(device)
    optimized_model = optimize_model(model).to(device)

    # Train the optimized model
    trained_model = train_optimized_model(optimized_model, train_loader, device, epochs=5)

    # Evaluate optimized model
    evaluate_model(trained_model, test_loader, device)

    # Export optimized model
    export_optimized_model(trained_model)

if __name__ == "__main__":
    main()
