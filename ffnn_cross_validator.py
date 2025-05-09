# %%
import warnings
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
# Based on the previous analysis, we know the data is stationary and we can use lag features
# From the previous analysis, we saw lag 16 was a good choice
test_set_lags = [8, 10 , 20, 30, 40, 50, 100]
hidden_layers = [1] #[1, 2, 3, 4]
initial_sizes = [256] #[16, 32, 64, 128, 256]
# We will now test the parameter combinations above with cross-validation and select the best one.
# Some of these combinations will not be reasonable, such as windows size = 8 and initial size = 128, but we will test them anyway,
# because data is small, training is fast and it is easier to test them all and ignore the bad ones later.

best_mse = float("inf")
best_model = None
best_combination = None
show_plots = False
iterations = len(test_set_lags) * len(hidden_layers) * len(initial_sizes)
current_iteration = 0
test_results = []

# %% Cross-validation for hyperparameter tuning. 
for window_size in test_set_lags:
    for num_hidden_layers in hidden_layers:
        for initial_size in initial_sizes:
            
            current_iteration += 1
            if initial_size <= window_size:
                # Skip this combination, it is not reasonable
                print(f"Skipping unreasonable combination: Window size: {window_size}, Hidden Layers: {num_hidden_layers}, Initial Size: {initial_size}.")
                continue    
            
            #   Another strategy to skip unreasonable combinations is skipping too deep models.
            # If window_size x hidden_layers >= initial_size
            # This will skip the combinations like 8,4,32 or 16,4,32, these are too deep for the input size.
            # We will omit this rule for windows 40 and 50, because they might be large enough to handle it.
            if window_size * num_hidden_layers > initial_size and window_size < 40:
                 print(f"Skipping too deep structure: Window size: {window_size}, Hidden Layers: {num_hidden_layers}, Initial Size: {initial_size}.")
                 continue
            
            # Current combination: lag={window_size}, hidden_layers={num_hidden_layers}, initial_size={initial_size}
            msg = f"Running Cross-Validation: Window size: {window_size}, Hidden Layers: {num_hidden_layers}, Initial Size: {initial_size}. Progress: {current_iteration}/{iterations}"
            print("-" * len(msg))
            print(msg)
            
            drop_out = 0.1 # Dropout rate for regularization
            pred_horizon = 200  # Number of time steps to predict
            epochs = int(300 * drop_out) # Number of epochs for CV training. Lower dropout means less epochs.

            # Build the model
            model = fn.build_ffnn_model(input_dim=window_size, num_hidden_layers=num_hidden_layers, initial_size=initial_size, drop_out=drop_out)
            #Get the number of tunable parameters in the model
            num_params = model.count_params()
            #model.summary()
            
            # Train the model and get the training history
            results = fn.train_cv(model=model, full_dataset=df, lag_order=window_size, epochs=epochs, enable_early_stopping=False)
            history = results["history"]

            # Calculate MSE on descaled values
            mse_cv_average = results["mse"]
            test_results.append((window_size, num_hidden_layers, initial_size, mse_cv_average, num_params))
            print(f"CV-MSE (avg): {mse_cv_average:.4f}, Window size: {window_size}, Hidden Layers: {num_hidden_layers}, Initial Size: {initial_size}. Progress: {current_iteration}/{iterations}")
            if mse_cv_average < best_mse:
                best_mse = mse_cv_average
                best_model = model
                best_combination = (window_size, num_hidden_layers, initial_size)
                print(f"New best model found with MSE: {best_mse:.4f}")
                # Save the model
                model.save("mert_best_model.h5")
            else:
                #print the current best model
                print(f"Current best model: MSE: {best_mse:.4f}, Window size: {best_combination[0]}, Hidden Layers: {best_combination[1]}, Initial Size: {best_combination[2]}")
            
            if show_plots:
                # Plot training history
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
                
                # Predict the next 200 time steps.
                epochs = 200 # We will first use early stopping to find the best epoch, then we will train with the full dataset.
                #drop_out = 0.2

                # Build the model and train with 80% of the data. This is just to approximate the correct epoch number for the current architecture.
                model = fn.build_ffnn_model(input_dim=window_size, num_hidden_layers=num_hidden_layers, initial_size=initial_size, drop_out=drop_out)
                #model.summary()
                # Train the model and get the training history
                results = fn.train(model=model, full_dataset=df, lag_order=window_size, epochs=epochs)
                history = results["history"]

                # Descale the predictions to get actual values
                future_predictions_descaled = fn.generate_future_predictions(model, window_size, pred_horizon)

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
            
            print("-" * len(msg))


# %% Print the best model and its parameters
print(f"Best model parameters: Window size: {best_combination[0]}, Hidden Layers: {best_combination[1]}, Initial Size: {best_combination[2]}")
print(f"Best model MSE: {best_mse:.4f}")
# Save the best model
best_model.save("mert_best_model.h5")
# Save the test results to a CSV file
test_results_df = pd.DataFrame(test_results, columns=["Window Size", "Hidden Layers", "Initial Size", "MSE","Tunable Parameters"])
test_results_df.to_csv("mert_test_results.csv", index=False)
#sort the results by MSE
test_results_df = test_results_df.sort_values(by="MSE")
# Print test results as table
print("Test results:")
print(test_results_df.to_string(index=False))

# %% Final training with the best model
epochs = 200 # We will first use early stopping to find the best epoch, then we will train with the full dataset.
#drop_out = 0.2

# Build the model and train with 80% of the data. This is just to approximate the correct epoch number for the current architecture.
model = best_model
window_size = best_combination[0]
#model.summary()
# Train the model and get the training history
results = fn.train(model=model, full_dataset=df, lag_order=window_size, epochs=epochs)
history = results["history"]

# Descale the predictions to get actual values
future_predictions_descaled = fn.generate_future_predictions(model, window_size, pred_horizon)

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
