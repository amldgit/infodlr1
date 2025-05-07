import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from mert_fnn import FNN
import torch.nn as nn

window_size = 30
batch_size = -1
epochs = 100
hidden_layers = 2
hidden_size = 64
shuffle = True
    
model = FNN(input_size=window_size, hidden_size=hidden_size, hidden_layers=hidden_layers)
model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
model.loss_fn = nn.MSELoss()
print(model)

model.perform_training(epochs=epochs, sequence_len=window_size,batch_size=batch_size, shuffle=shuffle)
results = model.get_descaled_results()
train_losses = np.array(results["train_losses"])
test_losses = np.array(results["test_losses"])
predictions = np.array(results["predictions"])
actual = np.array(results["actual"])
print("Final train loss: ", results["train_score"])
print("Final test loss: ", results["test_score"])

#load the best model
import mert_data as dt
model.load_state_dict(torch.load(model.path_best))
model.eval()
X,y = model.data_set[0],model.data_set[1]
# Initialize lists to store predictions
future_predictions = []
last_window = X[-1]  # Get the last window from training data

# Predict next 200 points
for _ in range(200):
    # Reshape the window for model input
    input_tensor = torch.FloatTensor(last_window).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        pred = model(input_tensor)
    
    # Append the prediction
    future_predictions.append(pred.item())
    
    # Update the window by removing first element and adding the prediction
    last_window = np.append(last_window[1:], pred.item())

# Convert predictions to numpy array
future_predictions = np.array(future_predictions)
# Descale the predictions
future_predictions = dt.descale(future_predictions)

# Plot the future predictions
plt.figure(figsize=(12, 6))
plt.plot(range(len(actual)), actual, label='Historical Data')
plt.plot(range(len(actual), len(actual) + 200), future_predictions, label='Future Predictions')
plt.legend()
plt.title('Time Series Prediction')
plt.show()

# # Plot training and test losses
# num_epochs = len(train_losses)
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
# plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')

# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Losses Over Time')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.plot(predictions, label='Predicted')
# plt.plot(actual, label='Actual')
# plt.legend()
# plt.show()

