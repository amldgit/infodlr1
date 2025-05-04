import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import mert_data as dt
from mert_data import laser_dataset
from mert_fnn import FNN

window_size = 5
X_train, y_train, X_test, y_test = dt.split_data(train_size=800, sequence_len=window_size, normalize=True)
X_Train = laser_dataset(X_train, y_train)
y_train = laser_dataset(y_train, y_train)
X_test = laser_dataset(X_test, y_test)
y_test = laser_dataset(y_test, y_test)

model = FNN(input_size=window_size, hidden_size=32, hidden_layers=3)
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.3)
# Initialize lists to store losses
train_losses = []
test_losses = []
best_loss = np.inf
best_epoch = 0
epochs = 100
batch_size = len(X_Train)//10

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    # Training step
    train_loader = torch.utils.data.DataLoader(X_Train, batch_size=batch_size, shuffle=True)
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Calculate average training loss
    train_loss = train_loss / len(X_Train)
    train_losses.append(train_loss)
    
    # Validation step
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in torch.utils.data.DataLoader(X_test, batch_size=batch_size):
            outputs = model(inputs.float())
            test_loss += criterion(outputs, targets.unsqueeze(1)).item()
    
    test_loss = test_loss / len(X_test)
    test_losses.append(test_loss)
    
    if test_loss < best_loss:
        best_loss = test_loss
        best_epoch = epoch
        save_path = 'fnn_best_model.pt'
        torch.save(model.state_dict(), save_path)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Training Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}')

#load the best model
model.load_state_dict(torch.load(save_path))
model.eval()

# Make predictions
with torch.no_grad():
    test_loader = torch.utils.data.DataLoader(X_test, batch_size=len(X_test))
    for inputs, _ in test_loader:
        predictions = model(inputs.float())
    
    predictions = predictions.numpy()
    actual = y_test.y

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(actual, label='Actual')
plt.plot(predictions, label='Predicted')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Predictions vs Actual Values')
plt.legend()
plt.grid(True)
plt.show()

# Plot training and test losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Losses')
plt.legend()
plt.grid(True)
plt.show()