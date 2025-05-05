# %%
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from mert_fnn import FNN
import torch.nn as nn
from itertools import product
import mert_data as dt

window_sizes = [8, 16, 32]
batch_sizes = [16, 32, 64]
epoch_runs = [100, 250, 200]
hidden_layers = [1, 2, 3, 4]
hidden_sizes = [16, 32, 64]
#learning_rates = [0.001, 0.01, 0.1]
shuffles = [True, False]

param_combinations = {
    'window_size': window_sizes,
    'batch_size': batch_sizes,
    'epochs': epoch_runs,
    'hidden_layers': hidden_layers,
    'hidden_size': hidden_sizes,
    'shuffle': shuffles
}

# Generate all possible combinations
all_combinations = [dict(zip(param_combinations.keys(), v)) for v in product(*param_combinations.values())]

# Print total number of combinations
print(f"Total combinations to test: {len(all_combinations)}")

# Example of how to use each combination
# %%
cnt = len(all_combinations)
current_combination = 0
best_combination = None
best_loss = float('inf')

for params in all_combinations:    
    current_combination += 1
    print(f"Combination {current_combination} of {cnt}")    
    window_size = params['window_size']
    batch_size = params['batch_size']
    epochs = params['epochs']
    hidden_layers = params['hidden_layers']
    hidden_size = params['hidden_size']
    shuffle = params['shuffle']
    
    # Initialize the model with the current combination of parameters
    model = FNN(input_size=window_size, hidden_size=hidden_size, hidden_layers=hidden_layers)
    model.report = False
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    model.loss_fn = nn.MSELoss()
    #print(model)

    model.perform_training(epochs=epochs, train_size=800, sequence_len=window_size,batch_size=batch_size, shuffle=shuffle)
    results = model.get_descaled_results()
    train_losses = np.array(results["train_losses"])
    test_losses = np.array(results["test_losses"])
    predictions = np.array(results["predictions"])
    actual = np.array(results["actual"])
    min_loss = min(test_losses)
    if min_loss < best_loss:
        best_loss = min_loss
        best_combination = params.copy()
        print(f"New best combination: {best_combination} with loss: {best_loss}")
    
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

    # Plot training and test losses    
    num_epochs = len(train_losses)
    # Create a figure with 2 rows, first row has 2 columns
    fig = plt.figure(figsize=(15, 10))
    # Add text for parameters and best loss at the top of the figure
    plt.figtext(0.02, 1.05, f'Parameters: {params}', fontsize=10, ha='left')
    plt.figtext(0.02, 1.02, f'Best test loss: {min(test_losses)}', fontsize=10, ha='left')
    
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
    
    # Third plot (Historical Data and Future Predictions), spans full width
    plt.subplot(2, 1, 2)
    plt.plot(range(len(full_data)), full_data, label='Historical Data')
    plt.plot(range(len(full_data), len(full_data) + 200), future_predictions, label='Future Predictions')
    plt.title('Historical Data and Future Predictions')
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    #plt.show()

print(f"Best combination: {best_combination} with loss: {best_loss}")
