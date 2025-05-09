## %%
import warnings

from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import ffnn_keras_cv as fn

# Load the .mat file as data frame
__mat_data__ = scipy.io.loadmat('Xtrain.mat')
# Convert to DataFrame, there is only one variable in the .mat file
df = pd.DataFrame(__mat_data__['Xtrain']) 

    
## %% Data preparation
# Based on the previous analysis, we know the data is stationary and we'll use lag features
# From the previous analysis, we saw lag 16 was a good choice

# Set the lag order - number of previous time steps to use as features
lag_order = 40
num_layers = 1 # Number of hidden layers
initial_size = 256 # Initial size of the first hidden layer
drop_out = 0.0 # No Dropout for prediction.
pred_horizon = 200  # Number of time steps to predict
epochs = 30 # Number of epochs for training

# Build the model
model = fn.build_ffnn_model(input_dim=lag_order, num_hidden_layers=num_layers, initial_size=initial_size, drop_out=drop_out)
model.summary()
best_model_file = "ffnn_final_model.h5"
model.load_weights(best_model_file)
model.summary()

#This is required to get the last window for recurrent prediction
data = fn.scaler.fit_transform(df.values)
model.scaled_data = data

# Get the predictions, the returned predictions are already descaled
predictions = fn.generate_future_predictions(model, lag_order, pred_horizon)

#load the test data
__mat_data__ = scipy.io.loadmat('Xtest.mat')
# Convert to DataFrame, there is only one variable in the .mat file
df_test = pd.DataFrame(__mat_data__['Xtest']).values
# convert future preductions to integer
predictions = predictions.astype(int).T

#Calculate the MSE on the test set
mse = mean_squared_error(df_test, predictions)
print(f"Test MSE (on original scale): {mse:.4f}")

# Plot actual vs predicted for validation set (descaled)
plt.figure(figsize=(14, 7))
plt.plot(df_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Actual vs Predicted Values (Original Scale), MSE: {:.4f}'.format(mse))
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# %%
