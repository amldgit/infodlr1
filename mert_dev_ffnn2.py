import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import mert_data as dt
from mert_data import laser_dataset
from mert_fnn import FNN

window_size = 5
epochs = 100
batch_size = 32
shuffle = True

X_train, y_train, X_test, y_test = dt.split_data(train_size=800, sequence_len=window_size, normalize=True)
X_Train = laser_dataset(X_train, y_train)
y_train = laser_dataset(y_train, y_train)
X_test = laser_dataset(X_test, y_test)
y_test = laser_dataset(y_test, y_test)

model = FNN(input_size=window_size, hidden_size=32, hidden_layers=3)
model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
model.loss_fn = nn.MSELoss()
print(model)

final_test_loss, y_test_np, predictions_np = model.perform_training(epochs=epochs, train_size=800, sequence_len=window_size, normalize=True, batch_size=batch_size, shuffle=shuffle)
train_losses = np.array(model.train_losses)
test_losses = np.array(model.test_losses)

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

import matplotlib.pyplot as plt
plt.plot(predictions_np[100:], label='Predicted')
plt.plot(y_test_np[100:], label='Actual')
plt.legend()
plt.show()