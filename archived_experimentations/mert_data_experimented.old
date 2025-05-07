import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import MinMaxScaler
import torch

class laser_dataset(torch.utils.data.Dataset):  
    
    def __init__(self, X,y):
        self.X = X
        self.y = y        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y 

# Create sequences
def create_sequences(data:np.array, sequence_len:int) -> tuple:
    """Generates memory-aware sequences of data for time series prediction.
    It generates sequences of length `sequence_len` from the input data by shifting each time step by one.

    Args:
        data (array-like): _description_
        sequence_len (int): _description_

    Returns:
        tuple: X (input sequences, np.array), y (target values, np.array)
    """
    X, y = [], []
    for i in range(len(data) - sequence_len):
        X.append(data[i:i+sequence_len,0])
        y.append(data[i+sequence_len,0])
    return np.array(X), np.array(y)

scaler = MinMaxScaler(feature_range=(0, 1))
def preprocess(sequence_len=5, scale=True)->tuple:
    """
    Loads Xtrain.mat and splits the data into training and testing sets.
    
    Parameters:    
    - sequence_len: This is the number of time steps to look back.
    - scale: Whether to scale the data using MinMaxScaler.
    
    Returns:
    - X, y: data, labels.   
    """    
    
    #The data is in a .mat file. Load it using scipy.io
    __mat_data__ = scipy.io.loadmat('Xtrain.mat')
    # Convert to DataFrame, there is only one variable in the .mat file
    __df__ = pd.DataFrame(__mat_data__['Xtrain']) 
    # Your raw data, assume normalized or scaled to [0, 1]
    data = np.array(__df__.to_numpy(), dtype=np.float32)  # shape (1000,)
    
    if scale:        
        data = scaler.fit_transform(data)
    
    X_train, y_train = create_sequences(data, sequence_len)       
    return X_train, y_train

def descale(data)->np.array:
    """
    Inverse transform the data using the scaler.
    
    Parameters:
    - data: The data to inverse transform.
    
    Returns:
    - The inverse transformed data.
    """
    
    if isinstance(data, np.ndarray) and len(data) == 0 or data is None:
        return data       
     
    if isinstance(data, float) and (data == np.inf or data == -np.inf or data == np.nan):
        return data
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    #If data is scalar, convert to 2D
    if data.ndim == 0:
        data = data.reshape(1, -1)
        
    # Check if data is 1D and reshape it to 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    if len(data) == 0:
        return data
    
    return scaler.inverse_transform(data)