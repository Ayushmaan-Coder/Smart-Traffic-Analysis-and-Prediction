import os
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Path to the .h5 file
file_path = "D:/Code/METR-LA.h5"

# Open the .h5 file and read data
with h5py.File(file_path, 'r') as f:
    dfaxis0 = f['/df/axis0'][:]      # Adjust key names if different
    dfaxis1 = f['/df/axis1'][:]          
    dfblock0_items = f['/df/block0_items'][:]
    dfblock0_values = f['/df/block0_values'][:]


# Print summary
print("Loaded graph with {} sensors".format(len(dfaxis0)))

df = pd.read_hdf('D:/Code/METR-LA.h5', key='df')
print("DataFrame shape:", df.shape)

data = df.values  # shape: (num_timesteps, num_sensors)
print("Raw data shape:", data.shape)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print("Data has been standardized and stored in 'data_scaled'.")

def normalize_adj(adj):
    """Symmetric normalization of the adjacency matrix."""
    adj = adj + np.eye(adj.shape[0])
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

adj_norm = normalize_adj(dfaxis1)
adj_norm = torch.tensor(adj_norm, dtype=torch.float32)

def create_dataset(data, lookback, horizon):
    """
    Create input-output sequences for forecasting.
    
    Parameters:
      - data: numpy array of shape (num_timesteps, num_sensors)
      - lookback: number of past time steps for input
      - horizon: forecast horizon
      
    Returns:
      - X: (samples, lookback, num_sensors, 1)
      - y: (samples, num_sensors) for horizon==1 or (samples, num_sensors, horizon) for horizon>1
    """
    X, y = [], []
    T = data.shape[0]
    for i in range(T - lookback - horizon + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+horizon])
    X = np.array(X)[..., np.newaxis]  # (samples, lookback, num_sensors, 1)
    y = np.array(y)  # for horizon>1, shape: (samples, horizon, num_sensors)
    if horizon == 1:
        y = y.squeeze(1)  # (samples, num_sensors)
    else:
        # Transpose so that y has shape (samples, num_sensors, horizon)
        y = np.transpose(y, (0, 2, 1))
    return X, y

def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    return mae, rmse, mape

class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch).squeeze(-1)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * X_batch.size(0)
        epoch_train_loss /= len(train_loader.dataset)
        
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                output = model(X_batch).squeeze(-1)
                loss = criterion(output, y_batch)
                epoch_val_loss += loss.item() * X_batch.size(0)
        epoch_val_loss /= len(val_loader.dataset)
        scheduler.step(epoch_val_loss)
        print("Epoch {}/{} -- Train Loss: {:.4f} | Val Loss: {:.4f}".format(
            epoch+1, epochs, epoch_train_loss, epoch_val_loss))
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(X_batch).squeeze(-1)
            loss = criterion(output, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            predictions.append(output.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    return test_loss, predictions, actuals

# Graph Convolution layer for Base DSTAGNN (no dropout)
class GraphConvolutionBase(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionBase, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return F.relu(output)

class DSTAGNNBase(nn.Module):
    def __init__(self, in_channels, spatial_hidden, temporal_hidden, forecast_horizon, num_nodes, adj):
        super(DSTAGNNBase, self).__init__()
        self.num_nodes = num_nodes
        self.register_buffer('adj', adj)
        self.gcn = GraphConvolutionBase(in_channels, spatial_hidden)
        self.lstm = nn.LSTM(input_size=spatial_hidden, hidden_size=temporal_hidden,
                            num_layers=1, batch_first=True)
        self.fc = nn.Linear(temporal_hidden, forecast_horizon)
    def forward(self, x):
        batch_size, lookback, num_nodes, in_channels = x.size()
        spatial_outputs = []
        for t in range(lookback):
            x_t = x[:, t, :, :]
            spatial_out = self.gcn(x_t, self.adj)
            spatial_outputs.append(spatial_out)
        spatial_features = torch.stack(spatial_outputs, dim=1)
        spatial_features = spatial_features.transpose(1, 2).contiguous().view(batch_size * num_nodes, lookback, -1)
        lstm_out, _ = self.lstm(spatial_features)
        lstm_last = lstm_out[:, -1, :]
        out = self.fc(lstm_last)
        out = out.view(batch_size, num_nodes, -1)
        return out

# Enhanced DSTAGNN: two-layer LSTM with dropout
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        support = self.dropout(support)
        output = torch.matmul(adj, support)
        return F.relu(output)

class DSTAGNNEnhanced(nn.Module):
    def __init__(self, in_channels, spatial_hidden, temporal_hidden, forecast_horizon, num_nodes, adj, dropout=0.3):
        super(DSTAGNNEnhanced, self).__init__()
        self.num_nodes = num_nodes
        self.register_buffer('adj', adj)
        self.gcn = GraphConvolution(in_channels, spatial_hidden, dropout=dropout)
        self.lstm = nn.LSTM(input_size=spatial_hidden,
                            hidden_size=temporal_hidden,
                            num_layers=2,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(temporal_hidden, forecast_horizon)
    def forward(self, x):
        batch_size, lookback, num_nodes, in_channels = x.size()
        spatial_outputs = []
        for t in range(lookback):
            x_t = x[:, t, :, :]
            spatial_out = self.gcn(x_t, self.adj)
            spatial_outputs.append(spatial_out)
        spatial_features = torch.stack(spatial_outputs, dim=1)
        spatial_features = spatial_features.transpose(1, 2).contiguous().view(batch_size * num_nodes, lookback, -1)
        lstm_out, _ = self.lstm(spatial_features)
        lstm_last = lstm_out[:, -1, :]
        out = self.fc(lstm_last)
        out = out.view(batch_size, num_nodes, -1)
        return out
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lookback = 12
batch_size = 64
epochs = 30
learning_rate = 0.0005

results = []

def run_experiment(model_class, model_name, forecast_horizon, model_params):
    print("\nRunning experiment: {} with horizon = {}".format(model_name, forecast_horizon))
    X_exp, y_exp = create_dataset(data_scaled, lookback, forecast_horizon)
    num_samples = X_exp.shape[0]
    train_size = int(num_samples * 0.7)
    val_size = int(num_samples * 0.1)
    
    X_train = X_exp[:train_size]
    y_train = y_exp[:train_size]
    X_val   = X_exp[train_size:train_size+val_size]
    y_val   = y_exp[train_size:train_size+val_size]
    X_test  = X_exp[train_size+val_size:]
    y_test  = y_exp[train_size+val_size:]
    
    train_loader = DataLoader(TrafficDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TrafficDataset(X_val, y_val), batch_size=batch_size)
    test_loader  = DataLoader(TrafficDataset(X_test, y_test), batch_size=batch_size)
    
    num_nodes = X_exp.shape[2]
    model = model_class(**model_params, forecast_horizon=forecast_horizon, num_nodes=num_nodes, adj=adj_norm)
    
    train_model(model, train_loader, val_loader, epochs, learning_rate, device)
    
    _, preds, actuals = evaluate_model(model, test_loader, device)
    mae, rmse, mape = compute_metrics(actuals, preds)
    
    plt.figure(figsize=(10,4))
    plt.plot(actuals[:100, 0], label='Actual')
    plt.plot(preds[:100, 0], label='Predicted')
    plt.title("{} | Horizon: {}".format(model_name, forecast_horizon))
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.show()
    
    return {"Model": model_name, "Horizon": forecast_horizon, "MAE": mae, "RMSE": rmse, "MAPE": mape}

# Hyperparameters for Enhanced DSTAGNN
enhanced_params = {
    "in_channels": 1,
    "spatial_hidden": 64,
    "temporal_hidden": 128,
    "dropout": 0.3
}
# Hyperparameters for Base DSTAGNN
base_params = {
    "in_channels": 1,
    "spatial_hidden": 32,
    "temporal_hidden": 64
}

# Experiment 1: Enhanced DSTAGNN with horizon = 3
res1 = run_experiment(DSTAGNNEnhanced, "DSTAGNN-Enhanced", forecast_horizon=3, model_params=enhanced_params)
results.append(res1)

# Experiment 2: Enhanced DSTAGNN with horizon = 6
res2 = run_experiment(DSTAGNNEnhanced, "DSTAGNN-Enhanced", forecast_horizon=6, model_params=enhanced_params)
results.append(res2)

# Experiment 3: Enhanced DSTAGNN with horizon = 12
res3 = run_experiment(DSTAGNNEnhanced, "DSTAGNN-Enhanced", forecast_horizon=12, model_params=enhanced_params)
results.append(res3)

# Experiment 4: Base DSTAGNN with horizon = 6
res4 = run_experiment(DSTAGNNBase, "DSTAGNN-Base", forecast_horizon=6, model_params=base_params)
results.append(res4)
results_df = pd.DataFrame(results)
print(results_df)
results_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}%"})

