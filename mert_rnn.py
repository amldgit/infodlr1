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
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)  # out: [batch_size, seq_len, hidden_size]
        out = self.fc(out[:, -1, :])  # Take the last time step's output
        return out.squeeze()

model = RNNPredictor()
num_epochs = 40

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses_per_epoch[-1]:.4f}")

# Plot training and test losses
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses_per_epoch, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Losses Over Time')
plt.legend()
plt.grid(True)
plt.show()

model.load_state_dict(torch.load('rnn_best_model.pt'))
model.eval()

x_test, y_test = next(iter(test_loader))
preds = model(x_test).cpu().detach().numpy()
actual = y_test.numpy()

import matplotlib.pyplot as plt
plt.plot(preds[:100], label='Predicted')
plt.plot(actual[:100], label='Actual')
plt.legend()
plt.show()