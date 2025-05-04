import torch
import numpy as np
import pandas as pd
import scipy.io
import torch
import matplotlib.pyplot as plt

#The data is in a .mat file. Load it using scipy.io
__mat_data__ = scipy.io.loadmat('Xtrain.mat')
# Convert to DataFrame, there is only one variable in the .mat file
__df__ = pd.DataFrame(__mat_data__['Xtrain']) 

# Your raw data, assume normalized or scaled to [0, 1]
data = np.array(__df__.to_numpy(), dtype=np.float32)  # shape (1000,)
data = data / 255.0  # Normalize to [0, 1]

# Hyperparameters
window_size = 20
train_size = 800

# Create sequences
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

X, y = create_sequences(data, window_size)
X_train, y_train = X[:train_size - window_size], y[:train_size - window_size]
X_test, y_test = X[train_size - window_size:], y[train_size - window_size:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train).squeeze(-1)  # shape (N, window_size)
y_train_tensor = torch.tensor(y_train).unsqueeze(1)  # shape (N, 1)
X_test_tensor = torch.tensor(X_test).squeeze(-1)
y_test_tensor = torch.tensor(y_test).unsqueeze(1)

import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, input_size):
        super(FNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)
    
model = FNN(input_size=window_size)
criterion = nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.02)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.2)
# Initialize lists to store losses
train_losses = []
test_losses = []
best_loss = np.inf
best_epoch = 0
epochs = 100

for epoch in range(epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    # Store the training loss
    train_losses.append(loss.item())
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_output = model(X_test_tensor)
        test_loss = criterion(test_output, y_test_tensor)
        test_losses.append(test_loss.item())
        
    # Check for improvement
    if test_loss < best_loss:
        best_loss = test_loss
        best_epoch = epoch
        # Save the model state
        torch.save(model.state_dict(), 'fnn_best_model.pt')
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    final_test_loss = criterion(predictions, y_test_tensor)
    print(f"Final Test MSE: {final_test_loss.item():.4f}")

baseline_preds = X_test_tensor[:, -1]
baseline_mse = torch.mean((baseline_preds - y_test_tensor.squeeze())**2)
print("Naive Last-Value Baseline MSE:", baseline_mse.item())

# Plot training and test losses
#convert to original scale
train_losses = np.array(train_losses)
test_losses = np.array(test_losses)
baseline_mse_array = np.full_like(train_losses, baseline_mse.item())
#train_losses = np.sqrt(train_losses) * 255.0
#test_losses = np.sqrt(test_losses) * 255.0
num_epochs = len(train_losses)
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.plot(range(1, num_epochs + 1), baseline_mse_array, '--', label='Baseline MSE')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Losses Over Time')
plt.legend()
plt.grid(True)
plt.show()

model.load_state_dict(torch.load('fnn_best_model.pt'))
model.eval()

# Generate predictions with the loaded model
with torch.no_grad():
    predictions = model(X_test_tensor)

# Convert tensors to numpy for plotting
y_test_np = y_test_tensor.numpy().squeeze(2)
#y_yest_np = np.sqrt(y_test_np)*255.0
predictions_np = predictions.numpy() 
#predictions_np = np.sqrt(predictions_np)*255.0

import matplotlib.pyplot as plt
plt.plot(predictions_np[100:], label='Predicted')
plt.plot(y_test_np[100:], label='Actual')
plt.legend()
plt.show()

pass