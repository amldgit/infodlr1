# %%
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt
import pandas as pd
import scipy
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np
import mert_ffnn_keras as fn

#This is the .mat file as data frame.
__mat_data__ = scipy.io.loadmat('Xtrain.mat')
    # Convert to DataFrame, there is only one variable in the .mat file
df = pd.DataFrame(__mat_data__['Xtrain']) 

# %% [markdown]
# ## Data Inspection
# We will inspect the data to understand its structure and characteristics in the Xtrain.mat file.
# It contains a single variable, which appears to be sequential data, like a time series. 
# See the visualization of the data below:
plt.figure(figsize=(12, 6))
plt.plot(df[0])
plt.title('Sequential Data from Xtrain.mat')
plt.xlabel('Measurement Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# %% [markdown]
# Check the data for stationarity, which has a stable mean and variance over time.
# It is important to have stationary data to predict the next value.
# Below is a visualisation of the rolling mean and standard deviation.
window_size = 20
rolling_mean = df[0].rolling(window=window_size).mean()
rolling_std = df[0].rolling(window=window_size).std()

plt.figure(figsize=(12, 6))
plt.plot(df[0], label='Original')
plt.plot(rolling_mean, label='Rolling Mean')
plt.plot(rolling_std, label='Rolling Std')
plt.legend()
plt.title('Rolling Statistics')
plt.grid(True)
plt.show()

# window_size = 7
# rolling_mean = df[0].rolling(window=window_size).mean()
# rolling_std = df[0].rolling(window=window_size).std()

# plt.figure(figsize=(12, 6))
# plt.plot(df[0], label='Original')
# plt.plot(rolling_mean, label='Rolling Mean')
# plt.plot(rolling_std, label='Rolling Std')
# plt.legend()
# plt.title('Rolling Statistics')
# plt.grid(True)
# plt.show()

# %% [markdown]
# Based on the observations, the data appears to be stationary around the mean, but the variance seems to fluctuate.
# Let's perform statistical tests to confirm the stationarity of the data.
# The Augmented Dickey-Fuller (ADF) test and the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test are commonly used for this purpose.

# %%
from statsmodels.tsa.stattools import adfuller, kpss

adf_p = adfuller(df[0])[1]
adf_p = round(adf_p, 4)
# ADF Test (null hypothesis: time series is non-stationary)
# If p-value < 0.05, we reject the null hypothesis (data is stationary)
if adf_p < 0.05:
    print(f"ADF test: The data is stationary (p-value = {adf_p:>.3} < 0.05)")
else:
    print(f"ADF test: The data is non-stationary (p-value = {adf_p} >= 0.05)")
    
# KPSS test (null hypothesis: time series is stationary)
# If p-value < 0.05, we reject the null hypothesis (data is non-stationary)
kpss_p = kpss(df[0])[1]
kpss_p = round(kpss_p, 4)
if kpss_p < 0.05:
    print(f"KPSS test: The data is non-stationary (p-value={kpss_p} < 0.05)")
else:
    print(f"KPSS test: The data is stationary (p-value={kpss_p} >= 0.05)")

# %% [markdown]
# Both tests indicate that the data is stationary, so we can proceed with the inspection, 
# no need for differencing or transformation. 

# ## Lag Order Selection
# In this type of data, the order of the data points is important, and each data point carries information
# about next data points. However, as the data moves forward, the data points become less relevant, so we need to
# find a balance between how many data points we should remember and how many we should forget.
# It tells us how many previous data points we should consider to predict the next data point.
# Choosing the right lag order is crucial for the performance when working with sequential data.

# We will use 2 methods to evaluate the lag order:

# 1. Information Criteria (AIC, BIC, HQIC)

# 2. Cross-validation (MSE)

# and compare the results of both methods.

# %% [markdown]
# ### 1. Information Criteria
# We perform auto-regression for each lag order and calculate different loss function for each lag order.

# AIC applies lighter penalty for more parameters than BIC, and HQIC is a compromise between.
# It is common to use AIC if prediction is the goal, BIC is used if a simpler model is desired.

def evaluate_lag_order(series, max_lag=100):
    # Function to evaluate different lag orders.
    # We picked the trend as 'c' because the data is fluctuating around a stable mean value.
    results = {}
    for lag in range(1, max_lag+1):
        model = AutoReg(series.dropna(), lags=lag, trend='c')
        model_fitted = model.fit()
        results[lag] = {
            'aic': model_fitted.aic,
            'bic': model_fitted.bic,
            'hqic': model_fitted.hqic
        }
    return results

# Evaluate different lag orders
max_lag = 100
lag_evaluation = evaluate_lag_order(df[0], max_lag=max_lag)

# Plot AIC, BIC, HQIC for different lag orders
aics = [lag_evaluation[i]['aic'] for i in range(1, max_lag+1)]
bics = [lag_evaluation[i]['bic'] for i in range(1, max_lag+1)]
hqics = [lag_evaluation[i]['hqic'] for i in range(1, max_lag+1)]
plt.figure(figsize=(12, 6))
plt.plot(range(1, max_lag+1), aics, label='AIC')
plt.plot(range(1, max_lag+1), bics, label='BIC')
plt.plot(range(1, max_lag+1), hqics, label='HQIC')
plt.axvline(x=8, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=16, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Lag Order')
plt.ylabel('Information Criteria')
plt.legend()
plt.title('Information Criteria for Different Lag Orders')
plt.grid(True)
plt.show()

# %% [markdown]
# Based on the visual inspection, it look like lag order 7 and 15 are good candidates.

# ### 2. Cross-validation (MSE)
# We will compare different lag order values by using a simple linear model for prediction 
# and compare the MSE loss function.
def time_series_cv(series, max_lag=20, n_splits=10):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = {}
    
    for lag in range(1, max_lag+1):
        mse = []
        for train_idx, test_idx in tscv.split(series):
            train = series.iloc[train_idx]
            test = series.iloc[test_idx]
            
            # Skip if test set is too small for this lag
            if len(test) <= lag:
                continue
            
            # Create lagged features
            X_train = np.array([train.iloc[i-lag:i].values for i in range(lag, len(train))])
            y_train = train.iloc[lag:].values
            
            X_test = np.array([test.iloc[i-lag:i].values for i in range(lag, len(test))])
            y_test = test.iloc[lag:].values
            
            # Simple linear model, Ordinary Least Squares
            model = sm.OLS(y_train, X_train).fit()
            predictions = model.predict(X_test)
            
            mse.append(mean_squared_error(y_test, predictions))
        
        # Only calculate mean if we have values
        if mse:
            mse_scores[lag] = np.mean(mse)
        else:
            # Handle case where no folds were valid for this lag
            mse_scores[lag] = float('nan')  # or some other value
    
    return mse_scores

# Get MSE scores for different lag orders
max_lag = 100
mse_scores_ols = time_series_cv(df[0], max_lag=max_lag, n_splits=5)
# print the min mse score and the corresponding lag order
min_mse_ols = min(mse_scores_ols.values())
best_lag_ols = min(mse_scores_ols, key=mse_scores_ols.get)

# %% [markdown]
print(f"Best lag order: {best_lag_ols} with MSE: {min_mse_ols:.4f}")

# Get the 3 lowest MSE values and their corresponding lags
sorted_lags = sorted(mse_scores_ols.items(), key=lambda x: x[1])
best_3_lags_ols = [lag for lag, _ in sorted_lags[:3]]

# Plot MSE for different lag orders
plt.figure(figsize=(12, 6))
plt.plot(list(mse_scores_ols.keys()), list(mse_scores_ols.values()), marker='o')

# Add vertical lines for the 3 best lags
for lag in best_3_lags_ols:
    plt.axvline(x=lag, color='r', linestyle='--', 
                label=f'Best Lag {best_3_lags_ols.index(lag)+1}: {lag}')

plt.xlabel('Lag Order')
plt.ylabel('Mean Squared Error')
plt.title('Cross-Validation MSE for Different Lag Orders')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ### 3. Neural Network Model Selection and Initial Setup
# The best practice is to use recurrent neural networks (RNN) for sequential data,
# but we decided to use a feedforward neural network (FFNN). Considering the simplicity of the data,
# we argue that a FFNN is sufficient for this task and will be easier to implement.
# We chose the Keras library to implement the FFNN model for the same reason, 
# it encapsulates and simplifies the implementation more than the alternatives such as pytorch.
# As FFNN architecture we decided to use a pyramidal structure, which is a common practice based on 
# our research. Initially, the model has 3 hidden layers with 128, 64 and 32 neurons and predicts one single
# value. This structure ensures that the model learns many features in the first layer and then reduces 
# the number of features, eliminating the less important ones. Dropout layers are added between the hidden layers
# to prevent overfitting. The dropout rate is set to 0.2, which is a common practice. 
# We decided to use Adam optimizer with fixed learning rate of 0.001, with maximum number of epochs is set to 200.
# This will ensure that the model has enough time to learn the data, considering that the data is not too large. 
# Early stopping is used to stop the training if there is no improvement in the validation loss for 20 epochs to avoid overfitting. 
# The batch size is set to 32, which is a common practice for FFNN models. We think that this is a good starting
# point for the model, but we will experiment with different hidden layers and sizes later. 
#
# ### 4. Neural Network Validation MSE for Different Lag Orders
# We'll evaluate different lag orders using our neural network model to determine the best lag order.
# We will use lag orders from 1 to 40, because cross validation with OLS before showed that there is 
# no improvement after 40. The dropout will be disabled for this test, 
# because we do not want to introduce randomness in the model. We will deal with overfitting later.
def evaluate_nn_lag_orders(series, max_lag=30):
    """
    Evaluate different lag orders using neural network model and record validation MSE.
    """
    val_mse_scores = {}
    best_model = None
    best_lag = None
    best_mse = float('inf')
    dropout = 0.0  # Disable dropout for this test
    
    # Convert series to DataFrame as required by the train function
    data_df = series.to_frame()
    
    for lag in range(1, max_lag+1):
        print(f"Testing lag order: {lag}")
        
        # Build the neural network model
        model = fn.build_ffnn_model(input_dim=lag, drop_out=dropout)
        
        # Train the model and get results
        results = fn.train(model=model, full_dataset=data_df, lag_order=lag)
        
        # Record the validation MSE
        val_mse = results["mse"]
        val_mse_scores[lag] = val_mse
        print(f"Lag {lag}: Validation MSE = {val_mse:.4f}")
        
        # Check if this is the best model so far
        if val_mse < best_mse:
            best_mse = val_mse
            best_lag = lag
            best_model = model
    
    return val_mse_scores, best_model, best_lag, best_mse

# Evaluate different lag orders with neural network
max_lag_nn = 16  # Adjust based on available computation time
nn_val_mse_scores, best_nn_model, best_nn_lag, best_nn_mse = evaluate_nn_lag_orders(
    df[0], max_lag=max_lag_nn)

# %%
print(f"Best Neural Network lag order: {best_nn_lag} with Validation MSE: {best_nn_mse:.4f}")

# remove the first lag from the plot. 
# Because it does not carry any information about past data points and too high. It confuses the plot.
nn_val_mse_scores.pop(1, None)

# Find the 3 lowest MSE values and their corresponding lags
sorted_lags = sorted(nn_val_mse_scores.items(), key=lambda x: x[1])
best_3_lags = [lag for lag, _ in sorted_lags[:3]]

# Plot validation MSE for different lag orders
plt.figure(figsize=(12, 6))
plt.plot(list(nn_val_mse_scores.keys()), list(nn_val_mse_scores.values()), marker='o', color='green')

# Add vertical lines for the 3 best lags
for lag in best_3_lags:
    plt.axvline(x=lag, color='r', linestyle='--', 
                label=f'Best Lag {best_3_lags.index(lag)+1}: {lag}')

plt.xlabel('Lag Order')
plt.ylabel('Validation MSE')
plt.title('Neural Network Validation MSE for Different Lag Orders')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Depending on our experimentation, the variation for neural network starts around lag order 7 for FFNN, 
# which can be explained by the overfitting of the model. Please recall that we disabled the dropout to avoid randomness. 
# But this still gives a good insight about the lag order when considered together with the information criteria 
# and OLS based analysis earlier. We decided to focus on the lag orders 7, 15, 20 and 28 for the final model, based
# on our experimentation and the results of the previous analysis.
# We will enable the dropout again and train the model with these lag orders, 10 times for each lag order and
# average the results, to pick the best lag order. 
selected_lags = [7, 15, 20, 28]


# %%
