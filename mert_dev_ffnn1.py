import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
import mert_data as dt

window_size = 5

X_train, y_train, X_test, y_test = dt.split_data(train_size=800, sequence_len=window_size, normalize=True)
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train) # shape (N, window_size)
y_train_tensor = torch.tensor(y_train)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(FNN, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size//2),
            # nn.ReLU(),            
            nn.Linear(hidden_size, 1))
    
    def forward(self, x):
        return self.layers(x)
    
model = FNN(input_size=window_size, hidden_size=64)
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
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
    loss = criterion(output, y_train_tensor.unsqueeze(1))
    loss.backward()
    optimizer.step()
    
    # Store the training loss
    train_losses.append(loss.item())
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_output = model(X_test_tensor)
        test_loss = criterion(test_output, y_test_tensor.unsqueeze(1))
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
    final_test_loss = criterion(predictions, y_test_tensor.unsqueeze(1))
    print(f"Final Test MSE: {final_test_loss.item():.4f}")

# Plot training and test losses
#convert to original scale
train_losses = np.array(train_losses)
test_losses = np.array(test_losses)

#train_losses = np.sqrt(train_losses) * 255.0
#test_losses = np.sqrt(test_losses) * 255.0
num_epochs = len(train_losses)
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')

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
y_test_np = y_test_tensor.numpy()
#y_yest_np = np.sqrt(y_test_np)*255.0
predictions_np = predictions.numpy() 
#predictions_np = np.sqrt(predictions_np)*255.0

import matplotlib.pyplot as plt
plt.plot(predictions_np[100:], label='Predicted')
plt.plot(y_test_np[100:], label='Actual')
plt.legend()
plt.show()

pass