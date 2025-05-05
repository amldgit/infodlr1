import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from mert_fnn import FNN
import torch.nn as nn

window_size = 8
batch_size = -1
epochs = 150
hidden_layers = 2
hidden_size = 32
shuffle = True
    
model = FNN(input_size=window_size, hidden_size=hidden_size, hidden_layers=hidden_layers)
model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
model.loss_fn = nn.MSELoss()
print(model)

model.perform_training(epochs=epochs, train_size=800, sequence_len=window_size,batch_size=batch_size, shuffle=shuffle)
results = model.get_descaled_results()
train_losses = np.array(results["train_losses"])
test_losses = np.array(results["test_losses"])
predictions = np.array(results["predictions"])
actual = np.array(results["actual"])

# Plot training and test losses
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

plt.plot(predictions, label='Predicted')
plt.plot(actual, label='Actual')
plt.legend()
plt.show()

import mert_data as dt
full_data = dt.get_dataset(scale=True)
# Get the last window from the dataset
last_window = full_data[-window_size:]
last_window = torch.FloatTensor(last_window).reshape(1, window_size)

# Predict next 200 points recursively
future_predictions = []
current_window = last_window

for _ in range(200):
    # Get prediction for next point
    with torch.no_grad():
        prediction = model(current_window.unsqueeze(0))
    
    # Add prediction to results
    future_predictions.append(prediction.item())
    
    # Update window by removing oldest value and adding prediction
    current_window = torch.cat((current_window[:, 1:], prediction.reshape(1, 1)), dim=1)

#descale full data and future predictions
#full_data = dt.descale(full_data)
#future_predictions = dt.descale(future_predictions)
# Plot the future predictions
plt.figure(figsize=(10, 6))
plt.plot(range(len(full_data)), full_data, label='Historical Data')
plt.plot(range(len(full_data), len(full_data) + 200), future_predictions, label='Future Predictions')
plt.title('Historical Data and Future Predictions')
plt.legend()
plt.show()