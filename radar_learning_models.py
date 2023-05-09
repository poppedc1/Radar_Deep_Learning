import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

# Define the feedforward linear model
class RadarFeedForward(nn.Module):
    def __init__(self, input_dim, target_dim, hidden_dim, num_layers, dropout):
        super(RadarFeedForward, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, target_dim))

        # Combine the layers using the Sequential container
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Define the LSTM model
class RadarLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_dim, num_layers):
        super(RadarLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, target_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

#Currently set to only use MSE loss, but we can enforce geometric constraint loss as well (doesn't improve accuracy much)
def custom_loss(targets, predictions, radar_array, N):
    mse_loss = torch.nn.MSELoss()

    # Original MSE loss
    loss1 = mse_loss(targets, predictions)

    # Additional MSE loss term for geometric constraints    
    batch_size = targets.shape[0]
    radar_array = torch.reshape(radar_array, (batch_size, N, 3))
    predictions = torch.reshape(predictions, (batch_size, N, 3))
    original_geometric_constraints = torch.cdist(radar_array, radar_array)
    predicted_geometric_constraints = torch.cdist(predictions, predictions)
    loss2 = mse_loss(original_geometric_constraints, predicted_geometric_constraints)

    # Combine the losses
    total_loss = loss1 #+ loss2

    return total_loss

#train
def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs, N):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, targets, radar_array = data
            inputs, targets, radar_array = inputs.to(device), targets.to(device), radar_array.to(device)

            optimizer.zero_grad()

            outputs = model(inputs.float())
            loss = custom_loss(targets.float(), outputs, radar_array.float(), N)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"epoch {epoch + 1}, loss: {running_loss / len(train_loader)}")

    print("training over")

#Test
def evaluate_model(model, test_loader, device):
    model.eval()
    model.to(device)

    predictions = []
    targets = []
    radar_arrays = []

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, target, radar_array = data
            inputs, target, radar_array = inputs.to(device), target.to(device), radar_array.to(device)
            output = model(inputs.float())

            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            radar_arrays.append(radar_array.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    radar_arrays = np.concatenate(radar_arrays, axis=0)

    return predictions, targets, radar_arrays