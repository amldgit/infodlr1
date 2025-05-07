import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import scipy.io
import torch
#from sklearn.preprocessing import StandardScaler

#The data is in a .mat file. Load it using scipy.io
__mat_data__ = scipy.io.loadmat('Xtrain.mat')
# Convert to DataFrame, there is only one variable in the .mat file
__df__ = pd.DataFrame(__mat_data__['Xtrain']) 

# Dummy dataset for example
data = __df__.to_numpy().T[0]
data = data / 255.0  # Normalize to [0, 1]

train_data = data[:800]
test_data = data[800:]

class SequenceDataset(Dataset):
    def __init__(self, data, seq_length=10):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length]
        return torch.tensor(x).unsqueeze(1), torch.tensor(y)

seq_length = 20
train_dataset = SequenceDataset(train_data, seq_length)
test_dataset = SequenceDataset(test_data, seq_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

import torch.nn as nn

class RNNPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(RNNPredictor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)  # out: [batch_size, seq_len, hidden_size]
        out = self.fc(out[:, -1, :])  # Take the last time step's output
        return out.squeeze()

model = RNNPredictor(input_size=1, hidden_size=32, num_layers=2)
num_epochs = 120

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Add plateau scheduler
scheduler = None
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=40)
# Counters:
best_loss = float('inf')

# Lists to store losses
train_losses = []
test_losses_per_epoch = []

# Modify training loop to track losses
for epoch in range(num_epochs):
    model.train()
    epoch_losses = []
    
    for x_batch, y_batch in train_loader:
        output = model(x_batch)
        loss = criterion(output, y_batch)
        epoch_losses.append(loss.item())
        if scheduler:            
            scheduler.step(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_losses.append(np.mean(epoch_losses))
    
    # Calculate test loss for this epoch
    model.eval()
    test_epoch_losses = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            output = model(x_batch)
            loss = criterion(output, y_batch)
            test_epoch_losses.append(loss.item())
    
    test_loss_avg = np.mean(test_epoch_losses)
    test_losses_per_epoch.append(test_loss_avg)
    
    if test_loss_avg < best_loss:
        best_loss = test_loss_avg
        # Save the model if needed
        torch.save(model.state_dict(), 'rnn_best_model.pt')
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses_per_epoch[-1]:.4f}, LR: {current_lr:.6f}")
#load the best model
model.load_state_dict(torch.load('rnn_best_model.pt'))
model.eval()
# Make predictions
with torch.no_grad():
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    for x_batch, _ in test_loader:
        predictions = model(x_batch.float())
        actual = test_data[seq_length:]
        final_test_loss = criterion(predictions, torch.tensor(actual).unsqueeze(1))
        final_test_loss = final_test_loss.item()
        print(f"Final Test Loss: {final_test_loss:.4f}")

        # Get the last sequence from test data for initial prediction
        last_sequence = test_data[-seq_length:]
        last_sequence = torch.FloatTensor(last_sequence).unsqueeze(1).unsqueeze(0)  # Shape: [1, seq_length, 1]

        # Generate future predictions
        future_predictions = []
        current_sequence = last_sequence.clone()

        for _ in range(200):
            # Get prediction for next value
            with torch.no_grad():
                next_pred = model(current_sequence).item()
            future_predictions.append(next_pred)
            
            # Update sequence for next prediction
            current_sequence = torch.cat([current_sequence[:, 1:, :], 
                                        torch.FloatTensor([next_pred]).unsqueeze(0).unsqueeze(1)], 
                                       dim=1)

        # Combine all data for plotting
        full_data = np.concatenate([train_data, test_data])

# Descale arrays for plotting
full_data = full_data * 255
train_losses = [loss * 255 for loss in train_losses]
test_losses_per_epoch = [loss * 255 for loss in test_losses_per_epoch]

# Convert MSE to MAE by taking square root
train_losses = [np.sqrt(loss) for loss in train_losses]
test_losses_per_epoch = [np.sqrt(loss) for loss in test_losses_per_epoch]
predictions = predictions.numpy() * 255
future_predictions = np.array(future_predictions) * 255
actual = actual * 255
final_test_loss = final_test_loss * 255
print(f"Final Test Loss (MAE): {np.sqrt(final_test_loss):.4f}")

# Plot training and test losses    
num_epochs = len(train_losses)
import matplotlib.pyplot as plt

# Create figure with subplots
fig = plt.figure(figsize=(14, 9))

# Plot losses
plt.subplot(2, 2, 1)
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.plot(range(num_epochs), test_losses_per_epoch, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Losses (MAE)')
plt.legend()

# Plot actual vs predicted values
plt.subplot(2, 2, 2)
plt.plot(actual, label='Actual')
plt.plot(predictions, label='Predicted')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.legend()

# Plot full dataset and future predictions
plt.subplot(2, 1, 2)
plt.plot(full_data, label='Historical Data')
plt.plot(range(len(full_data), len(full_data) + len(future_predictions)), 
         future_predictions, label='Future Predictions', color='red')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Full Dataset with Future Predictions')
plt.legend()

plt.tight_layout()
plt.show()