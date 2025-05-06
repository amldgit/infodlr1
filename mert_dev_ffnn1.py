import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from mert_fnn import FNN
import torch.nn as nn

window_size = 7
batch_size = 16
epochs = 100
hidden_layers = 4
hidden_size = 64
shuffle = True
    
model = FNN(input_size=window_size, hidden_size=hidden_size, hidden_layers=hidden_layers)
model.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#model.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer, mode='min', factor=0.1, patience=80)
model.loss_fn = nn.MSELoss()
model.report_interval = epochs // 10
print(model)

model.perform_training(epochs=epochs, sequence_len=window_size,batch_size=batch_size, shuffle=shuffle)
results = model.get_descaled_results()
train_losses = np.array(results["train_losses"])
test_losses = np.array(results["test_losses"])
predictions = np.array(results["predictions"])
actual = np.array(results["actual"])
print("Train loss average: ", np.mean(results["train_losses"]))
print("Final test loss: ", results["final_test_loss"])

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

# Plot training and test losses    
num_epochs = len(train_losses)
# Create a figure with 2 rows, first row has 2 columns
fig = plt.figure(figsize=(14, 9))    
# First plot (Training and Test Losses)
plt.subplot(2, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Losses Over Time')
plt.legend()
plt.grid(True)

# Second plot (Predictions vs Actual)
plt.subplot(2, 2, 2)
plt.plot(predictions, label='Predicted')
plt.plot(actual, label='Actual')
plt.title('Predictions vs Actual')
plt.legend()

# Third plot (Historical Data and Future Predictions)
plt.subplot(2, 1, 2)
plt.plot(range(len(full_data)), full_data, label='Historical Data', color='blue')
plt.plot(range(len(full_data), len(full_data) + len(future_predictions)), 
            future_predictions, label='Future Predictions', color='red')
plt.axvline(x=len(full_data), color='gray', linestyle='--')
plt.title('Historical Data and Future Predictions')
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

