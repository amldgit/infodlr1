import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from mert_fnn import FNN
import torch.nn as nn

window_size = 10
batch_size = -1
epochs = 100
hidden_layers = 3
hidden_size = 32
shuffle = True
    
model = FNN(input_size=window_size, hidden_size=hidden_size, hidden_layers=hidden_layers)
model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
model.loss_fn = nn.MSELoss()
print(model)

model.perform_training(epochs=epochs, train_size=-1, sequence_len=window_size,batch_size=batch_size, shuffle=shuffle)
results = model.get_descaled_results()
train_losses = np.array(results["train_losses"])
#test_losses = np.array(results["test_losses"])
#predictions = np.array(results["predictions"])
#actual = np.array(results["actual"])
print("Final train loss: ", results["train_score"])
#print("Final test loss: ", results["test_score"])

# Plot training and test losses
num_epochs = len(train_losses)
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
#plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Losses Over Time')
plt.legend()
plt.grid(True)
plt.show()

# plt.plot(predictions, label='Predicted')
# plt.plot(actual, label='Actual')
# plt.legend()
# plt.show()

