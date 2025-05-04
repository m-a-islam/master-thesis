from typing import Tuple, Union, Optional

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

column_names = ['unit_number', 'time_in_cycles'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in
                                                                                               range(1, 24)]


def load_dataframe(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        sep=' ',
        header=None,
        names=column_names
    )
    return df


def clean_data(df: pd.DataFrame) -> list:
    removed_columns = []

    # Remove empty columns
    empty_cols = df.columns[df.isna().all()]
    removed_columns.extend(empty_cols)

    # Remove columns with a standard deviation smaller than 0.02
    low_std_cols = [col for col in df.columns if ('sensor' in col or 'op_setting' in col) and df[col].std() < 0.02]
    removed_columns.extend(low_std_cols)

    print("Removed columns:", removed_columns)
    return removed_columns


# Calculate Remaining Useful Life (RUL)
def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    df = df.merge(max_cycles, on='unit_number', how='left')
    df['RUL'] = df['max_cycle'] - df['time_in_cycles']
    df.drop(columns=['max_cycle'], inplace=True)
    return df


# Normalize sensor data
def normalize_data(df: pd.DataFrame, scaler: MinMaxScaler = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    # Extract sensor columns
    sensor_cols = [col for col in df.columns if 'sensor' in col or 'op_setting' in col]
    if scaler is None:
        scaler = MinMaxScaler()
        df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    else:
        df[sensor_cols] = scaler.transform(df[sensor_cols])
    return df, scaler


def prep_data(file_path: str) -> Tuple[pd.DataFrame, MinMaxScaler]:
    df = load_dataframe(file_path)
    df = add_rul(df)
    df, scaler = normalize_data(df, scaler=None)
    return df, scaler


# Prepare sequences for LSTM input
def prepare_train_sequences(df: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    sensor_cols = [col for col in df.columns if 'sensor' in col or 'op_setting' in col]
    X, y = [], []
    for unit in df['unit_number'].unique():
        unit_data = df[df['unit_number'] == unit]
        for i in range(len(unit_data) - sequence_length):
            X.append(unit_data[sensor_cols].iloc[i:i + sequence_length].values)
            y.append(unit_data['RUL'].iloc[i + sequence_length])
    X, y = np.array(X), np.array(y)

    return X, y


def prepare_test_sequences(test_data: pd.DataFrame, test_RUL: pd.DataFrame, sequence_length: int) -> Tuple[
    np.ndarray, np.ndarray]:
    sensor_cols = [col for col in test_data.columns if 'sensor' in col or 'op_setting' in col]
    X, y = [], []
    for unit in test_data['unit_number'].unique():
        unit_data = test_data[test_data['unit_number'] == unit]
        if unit_data.shape[0] > sequence_length:
            X.append(unit_data[sensor_cols].iloc[-sequence_length:].values)
            y.append(test_RUL['RUL'].iloc[unit - 1])
    X, y = np.array(X), np.array(y)

    return X, y


class MLPmodel(nn.Module):
    def __init__(self, layer_units: list, input_size: int, output_size: int, dropout_rate: float = 0.5):
        super(MLPmodel, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.model_type = 'MLP'

        # Add the first layer
        self.layers.append(nn.Linear(input_size, layer_units[0]))
        self.dropouts.append(nn.Dropout(dropout_rate))

        # Add hidden layers
        for i in range(1, len(layer_units)):
            self.layers.append(nn.Linear(layer_units[i - 1], layer_units[i]))
            self.dropouts.append(nn.Dropout(dropout_rate))

        # Add the output layer (no dropout here)
        self.layers.append(nn.Linear(layer_units[-1], output_size))

        # Initialize weights
        self.init_weights()

    def init_weights(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')  # Kaiming for ReLU
                nn.init.constant_(layer.bias, 0)  # Bias initialized to 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = nn.ReLU()(x)
            x = self.dropouts[i](x)  # Apply dropout

        # Output layer (no activation or dropout)
        x = self.layers[-1](x)
        return x


class LSTMmodel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout_rate: float = 0.5):
        super(LSTMmodel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = 'LSTM'

        # Define LSTM layer with dropout (applies dropout between layers if num_layers > 1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

        # Define a fully connected layer to map LSTM output to the target
        self.fc = nn.Linear(hidden_size, output_size)

        # Define a dropout layer before the fully connected layer
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        self.init_weights()

    def init_weights(self) -> None:
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # Hidden-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)  # Bias initialized to 0

        # Initialize fully connected layer weights
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Take the output of the last time step
        out = out[:, -1, :]

        # Apply dropout before the fully connected layer
        out = self.dropout(out)

        # Pass through the fully connected layer
        out = self.fc(out)
        return out


class CMAPSSDataset(Dataset):
    def __init__(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
        if not isinstance(X, (np.ndarray, torch.Tensor)):
            print("Warning: X is not a numpy array or torch.Tensor.")
        if not isinstance(y, (np.ndarray, torch.Tensor)):
            print("Warning: y is not a numpy array or torch.Tensor.")

        self.X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# Early stopping class
class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 0, path: str = "model_checkpoints/best_model.pth"):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            delta (float): Minimum change to qualify as an improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model: nn.Module) -> None:
        """Save the model when validation loss improves."""
        torch.save(model.state_dict(), self.path)


def train_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                initial_lr: float = 0.01, num_epochs: int = 5) -> Tuple[list, list, list]:
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create PyTorch Dataset and DataLoader
    train_dataset = CMAPSSDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Validation dataset
    val_dataset = CMAPSSDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    # Define the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=50, delta=0.0001)

    # Initialize lists to store loss values
    train_losses = []
    val_losses = []
    lr = []

    # Training loop with early stopping
    model.train()

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        model.train()  # Set model to training mode
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # Compute average training loss for the epoch
        train_losses.append(epoch_train_loss / len(train_loader))

        # Validation loss
        model.eval()  # Set model to evaluation mode
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                val_outputs = model(X_val_batch)
                val_loss = criterion(val_outputs.squeeze(), y_val_batch)
                epoch_val_loss += val_loss.item()

        # Compute average validation loss for the epoch
        val_losses.append(epoch_val_loss / len(val_loader))

        # Print progress with learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr.append(current_lr)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, LR: {current_lr:.6f}, ES Counter: {early_stopping.counter}, best loss: {early_stopping.best_loss:.4f}")

        # Step the scheduler with validation loss
        scheduler.step(val_losses[-1])

        # Check early stopping
        early_stopping(val_losses[-1], model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    return train_losses, val_losses, lr


def plot_training_progress(train_losses: list, val_losses: list, lr: list, save_path: Optional[str] = None) -> None:
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

    # Upper plot: Training and Validation Loss
    ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    ax1.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    ax1.set_yscale('log')  # Set y-axis to log scale
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss Over Epochs')
    ax1.legend()
    ax1.grid()

    # Lower plot: Learning Rate
    ax2.plot(range(1, len(lr) + 1), lr, label='Learning Rate', color='orange')
    ax2.set_yscale('log')  # Set y-axis to log scale
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Learning Rate')
    ax2.grid()

    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
    else:
        plt.show()


def predict_with_onnx(model_path: str, X: np.ndarray) -> np.ndarray:
    # Load the ONNX model
    session = ort.InferenceSession(model_path)

    # Get the input name for the ONNX model
    input_name = session.get_inputs()[0].name

    # Get the batch size from the ONNX model
    batch_size = session.get_inputs()[0].shape[0]

    # Split the data into batches
    num_samples = X.shape[0]
    predictions = []

    for i in range(0, num_samples, batch_size):
        batch = X[i:i + batch_size]
        # Run the model on the batch
        batch_predictions = session.run(None, {input_name: batch.astype(np.float32)})
        predictions.append(batch_predictions[0])

    # Combine all predicted batches into one output array
    y = np.concatenate(predictions, axis=0)

    return y
