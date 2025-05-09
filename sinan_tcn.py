# Install dependencies before running:
# pip install numpy scipy scikit-learn tensorflow keras matplotlib

import os
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Use standalone Keras or fall back to tf.keras
try:
    import keras
    from keras.layers import Input, Conv1D, Dense, Add, Activation, Flatten
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping
except ImportError:
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Conv1D, Dense, Add, Activation, Flatten
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

# 1. Load and preprocess the data
script_dir = os.path.dirname(os.path.abspath(__file__))
mat_path = os.path.join(script_dir, 'Xtrain.mat')
mat = loadmat(mat_path)
series = mat['Xtrain'].reshape(-1)
N = len(series)

# Plot raw vs scaled series
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1,1)).flatten()
plt.figure(figsize=(8,4))
plt.plot(series, color='steelblue', alpha=0.6, label='raw')
plt.plot(series_scaled * (series.max()-series.min()) + series.min(),
         color='orange', alpha=0.6, label='scaled (rescaled back)')
plt.title('Xtrain')
plt.legend()
plt.show()

# Utility: create windowed dataset
def create_dataset(series, window_size):
    Xs, ys = [], []
    for i in range(len(series) - window_size):
        Xs.append(series[i:i+window_size])
        ys.append(series[i+window_size])
    return np.array(Xs)[...,None], np.array(ys)

# 2. Build TCN block and model builder
def tcn_block(x, filters, kernel_size, dilation_rate):
    prev = x
    x = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
    x = Activation('relu')(x)
    x = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
    x = Activation('relu')(x)
    if prev.shape[-1] != filters:
        prev = Conv1D(filters, 1, padding='same')(prev)
    return Add()([prev, x])


def build_tcn(window_size, num_filters=32, kernel_size=3, dilations=[1,2,4,8]):
    inp = Input(shape=(window_size,1))
    x = inp
    for d in dilations:
        x = tcn_block(x, num_filters, kernel_size, d)
    x = Flatten()(x)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

# 3. Hyperparameter sweep: input lengths
input_lengths = [10,20,30,50,100,250,500]
train_mse_list = []
val_mse_list = []
for L in input_lengths:
    X, y = create_dataset(series_scaled, L)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    m = build_tcn(L)
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = m.fit(X_tr,y_tr, epochs=20, batch_size=64,
                    validation_data=(X_val,y_val), callbacks=[es], verbose=0)
    train_mse_list.append(min(history.history['loss']))
    val_mse_list.append(min(history.history['val_loss']))

# Plot MSE vs input length
plt.figure(figsize=(8,4))
plt.plot(input_lengths, train_mse_list, marker='o', label='Train MSE')
plt.plot(input_lengths, val_mse_list, marker='x', linestyle='--', label='Val MSE')
plt.xscale('log')
plt.xlabel('Input Length')
plt.ylabel('MSE')
plt.title('TCN - MSE vs Input Length')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.show()

# 4. Train final TCN model with chosen window_size
window_size = 20
X_all, y_all = create_dataset(series_scaled, window_size)
X_train, X_eval, y_train, y_eval = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42)
model = build_tcn(window_size)
model.fit(X_train, y_train, epochs=50, batch_size=32,
          validation_split=0.2,
          callbacks=[EarlyStopping(monitor='val_loss',patience=5)], verbose=1)

# 5. Single-step eval: actual vs predicted (scaled)
y_pred_s = model.predict(X_eval)
y_true_s = y_eval.reshape(-1,1)
plt.figure(figsize=(6,4))
plt.plot(y_true_s[:200], label='actual')
plt.plot(y_pred_s[:200], label='predicted')
plt.title('Predictions')
plt.legend()
plt.show()

# 6. Multi-step recursive forecast
horizon = 200
start = N - window_size - horizon
window = list(series_scaled[start:start+window_size])
preds_scaled = []
for _ in range(horizon):
    x_in = np.array(window[-window_size:])[None,:,None]
    next_s = float(model.predict(x_in))
    preds_scaled.append(next_s)
    window.append(next_s)

true_future = series[start+window_size:start+window_size+horizon]
preds = scaler.inverse_transform(np.array(preds_scaled)[:,None]).flatten()

# Plot original series and forecast
plt.figure(figsize=(10,4))
plt.plot(np.arange(N), series, color='gray', alpha=0.4, label='Original signal')
plt.plot(np.arange(start+window_size, start+window_size+horizon), true_future,
         color='blue', label='True')
plt.plot(np.arange(start+window_size, start+window_size+horizon), preds,
         color='orange', label='Predicted')
plt.title(f'Input Length = {window_size} | MSE = {mean_squared_error(true_future, preds):.4f}')
plt.legend()
plt.show()
