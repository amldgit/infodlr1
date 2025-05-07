import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def build_ffnn_model(input_dim, num_hidden_layers=2, initial_size=128, drop_out=0.2):
    layers = []
    # First layer
    layers.append(Dense(initial_size, activation='relu', input_dim=input_dim))
    if drop_out > 0:
        layers.append(Dropout(drop_out))
    
    # Hidden layers
    current_size = initial_size
    for _ in range(num_hidden_layers):        
        #current_size = current_size
        layers.append(Dense(current_size, activation='relu'))
        current_size = current_size // 2
        if drop_out > 0:
            layers.append(Dropout(drop_out))
    
    # Output layer
    layers.append(Dense(1))  # Output layer for regression
    
    model = Sequential(layers)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
# Create the dataset with lag features
def create_dataset(data, lag):    
    """Create a dataset with lag features."""
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i-lag:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

batch_size = 16
#scaled_data = None
def train_full(model, full_dataset:pd.DataFrame, lag_order=7, epochs=200) -> dict:    
    #global scaled_data
    data = scaler.fit_transform(full_dataset.values)
    model.scaled_data = data
    
    # Scale the data and split to sequences, each sequence is lag_order long.
    X, y = create_dataset(data, lag_order)   

    early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True)
    
    execution_history = model.fit(
    X, y,
    #validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    #callbacks=[early_stopping],
    verbose=0)
    
    results = {"history": execution_history}
    # #Perform predictions
    # y_pred_scaled = model.predict(X_val)
    
    # # Descale predictions and actual values as instructed.
    # y_pred_descaled = scaler.inverse_transform(y_pred_scaled).flatten()
    # results["predictions"] = y_pred_descaled
    
    # y_val_descaled = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    # results["actual"] = y_val_descaled
    
    # mse = mean_squared_error(y_val_descaled, y_pred_descaled)
    # results["mse"] = mse
    return results

#scaled_data = None
def train_cv(model, full_dataset:pd.DataFrame, lag_order=7, epochs=200, enable_early_stopping = False) -> dict:    
    #global scaled_data
    data = scaler.fit_transform(full_dataset.values)
    model.scaled_data = data
    
    # Scale the data and split to sequences, each sequence is lag_order long.
    X, y = create_dataset(data, lag_order)    
    tscv = TimeSeriesSplit(n_splits=5)

    cnt = 0
    losses = []
    predictions = []
    actuals = []
    training_losses = []
    validation_losses = []
    
    for train_index, test_index in tscv.split(X):
        cnt += 1
        print(f"/nFold {cnt}:")
        
        X_train, X_test = X[train_index], X[test_index]    
        y_train, y_test = y[train_index], y[test_index]
        
        if enable_early_stopping:
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)        
            execution_history = model.fit(X_train, y_train,validation_data=(X_test, y_test),epochs=epochs,batch_size=batch_size,callbacks=[early_stopping], verbose=0)
        else:  
            execution_history = model.fit(X_train, y_train,validation_data=(X_test, y_test),epochs=epochs,batch_size=batch_size, verbose=0)
        
        train_loss = np.array(execution_history.history['loss'])
        train_loss_descaled = scaler.inverse_transform(train_loss.reshape(-1, 1)).flatten()
        training_losses.append(train_loss_descaled)
        
        val_loss = np.array(execution_history.history['val_loss'])
        val_loss_descaled = scaler.inverse_transform(val_loss.reshape(-1, 1)).flatten()
        validation_losses.append(val_loss_descaled) 
        
        #Perform predictions
        y_pred_scaled = model.predict(X_test)
        
        # Descale predictions and actual values as instructed.
        y_pred_descaled = scaler.inverse_transform(y_pred_scaled).flatten()        
        predictions.append(y_pred_descaled)
        
        y_val_descaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        #results["actual"] = y_val_descaled
        actuals.append(y_val_descaled)
        
        mse = mean_squared_error(y_val_descaled, y_pred_descaled)
        #results["mse"] = mse
        losses.append(mse)
        print (f"Fold {cnt} - MSE: {mse}")
        
    # Calculate average MSE across all folds
    avg_mse = np.mean(losses)
    print(f"Average MSE across all folds: {avg_mse}")
    # Calculate average loss across all folds
    # Calculate average loss across all folds
    
    # Stack the arrays and compute mean along axis 0 (across folds)    
    if enable_early_stopping:
        avg_loss_train = None
        avg_loss_validation = None
    else:        
        avg_loss_train = np.mean(np.array(training_losses), axis=0)
        avg_loss_validation = np.mean(np.array(validation_losses), axis=0)
        
    #print(f"Average loss across all folds: {avg_loss_validation}")
    # Calculate average predictions across all folds
    avg_predictions = np.mean(predictions, axis=0)
    # Calculate average actuals across all folds
    avg_actuals = np.mean(actuals, axis=0)
    # Calculate average history across all folds
    #avg_loss_train = np.mean(training_losses, axis=0)
    
    history = {"loss":avg_loss_train , "val_loss": avg_loss_validation}
    results = {"history":history, "predictions": avg_predictions, "actual": avg_actuals, "mse": avg_mse}
    return results
    
def generate_future_predictions(model, lag_order, n_steps)-> np.ndarray:
    """Predicts next n_steps data points using the trained model. 
    model.scaled_data must be set, this is automatically set in the train function.

    Args:
        model (_type_): trained model
        lag_order (_type_): sequence length
        n_steps (_type_): data points to predict

    Returns:
        np.ndarray: Array of predicted values. The values are descaled.
    """
    last_sequence = model.scaled_data[-lag_order:, 0]
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_steps):
        # Reshape the sequence for prediction
        current_input = current_sequence.reshape(1, -1)
        # Get the next predicted value
        next_pred = model.predict(current_input, verbose=0)[0, 0]
        # Append the prediction
        future_predictions.append(next_pred)
        # Update the sequence by removing the first element and adding the prediction
        current_sequence = np.append(current_sequence[1:], next_pred)
    
    future_predictions_scaled = np.array(future_predictions)
    future_predictions_descaled = scaler.inverse_transform(future_predictions_scaled.reshape(-1, 1)).flatten()
    return future_predictions_descaled