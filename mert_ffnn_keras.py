import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def build_ffnn_model(input_dim, num_layers=3, initial_size=128, drop_out=0.2):
    layers = []
    # First layer
    layers.append(Dense(initial_size, activation='relu', input_dim=input_dim))
    if drop_out > 0:
        layers.append(Dropout(drop_out))
    
    # Hidden layers
    current_size = initial_size
    for _ in range(num_layers - 2):  # -2 because we already have first and will add output
        current_size = current_size // 2
        layers.append(Dense(current_size, activation='relu'))
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

#scaled_data = None
def train(model, full_dataset:pd.DataFrame, lag_order) -> dict:    
    #global scaled_data
    data = scaler.fit_transform(full_dataset.values)
    model.scaled_data = data
    
    # Scale the data and split to sequences, each sequence is lag_order long.
    X, y = create_dataset(data, lag_order)
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True)
    
    execution_history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=0)
    
    results = {"history": execution_history}
    #Perform predictions
    y_pred_scaled = model.predict(X_val)
    
    # Descale predictions and actual values as instructed.
    y_pred_descaled = scaler.inverse_transform(y_pred_scaled).flatten()
    results["predictions"] = y_pred_descaled
    
    y_val_descaled = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    results["actual"] = y_val_descaled
    
    mse = mean_squared_error(y_val_descaled, y_pred_descaled)
    results["mse"] = mse
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