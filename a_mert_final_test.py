# %%
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import a_mert_ffnn_keras_cv as fn

# Load the .mat file as data frame
__mat_data__ = scipy.io.loadmat('Xtrain.mat')
# Convert to DataFrame, there is only one variable in the .mat file
df = pd.DataFrame(__mat_data__['Xtrain']) 

# %% Data preparation
# Based on the previous analysis, we know the data is stationary and we'll use lag features
# From the previous analysis, we saw lag 16 was a good choice

# Set the lag order - number of previous time steps to use as features
lag_order = 16
num_layers = 2 # Number of hidden layers
initial_size = 64 # Initial size of the first hidden layer
drop_out = 0.2 # Dropout rate for regularization
pred_horizon = 200  # Number of time steps to predict
epochs = 100 # Number of epochs for training

# Build the model
model = fn.build_ffnn_model(input_dim=lag_order, num_layers=num_layers, initial_size=initial_size, drop_out=drop_out)
#model.summary()

# Train the model and get the training history
results = fn.train(model=model, full_dataset=df, lag_order=lag_order, epochs=epochs, enable_early_stopping=False)
history = results["history"]

# %% Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Get the descaled actual and predicted values
y_val_descaled =  results["actual"]
y_pred_descaled = results["predictions"]

# Calculate MSE on descaled values
val_mse = results["mse"]
print(f"Validation MSE (on original scale): {val_mse:.4f}")

# Plot actual vs predicted for validation set (descaled)
plt.figure(figsize=(14, 7))
plt.plot(y_val_descaled, label='Actual')
plt.plot(y_pred_descaled, label='Predicted')
plt.title('Actual vs Predicted Values - Validation Set (Original Scale)')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Descale the predictions to get actual values
future_predictions_descaled = fn.generate_future_predictions(model, lag_order, pred_horizon)

# Get the last portion of historical data for plotting context (descaled)
#historical_data = df.iloc[-100:, 0].values
historical_data = df.values

# Plot the future predictions with historical data (all descaled)
plt.figure(figsize=(14, 7))
# Plot some historical data for context
plt.plot(range(-len(historical_data), 0), historical_data, label='Historical Data')
# Plot the predictions
plt.plot(range(0, pred_horizon), future_predictions_descaled, label='Predicted Future Values', color='red')
plt.axvline(x=0, color='k', linestyle='--')
plt.title(f'Prediction of Next {pred_horizon} Time Steps')
plt.xlabel('Time Steps')
plt.ylabel('Value (Original Scale)')
plt.legend()
plt.grid(True)
plt.show()

# %%
