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
def create_sequences(data, sequence_len):
    X, y = [], []
    for i in range(len(data) - sequence_len):
        X.append(data[i:i+sequence_len,0])
        y.append(data[i+sequence_len,0])
    return np.array(X), np.array(y)

def split_data(train_size=800, sequence_len=5, normalize=True):
    """
    Loads Xtrain.mat and splits the data into training and testing sets.
    
    Parameters:    
    - train_size: The number of samples to use for training.
    - sequence_len: The length of the sequences to create.
    
    Returns:
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Testing data and labels.
    """    
    
    #The data is in a .mat file. Load it using scipy.io
    __mat_data__ = scipy.io.loadmat('Xtrain.mat')
    # Convert to DataFrame, there is only one variable in the .mat file
    __df__ = pd.DataFrame(__mat_data__['Xtrain']) 
    # Your raw data, assume normalized or scaled to [0, 1]
    data = np.array(__df__.to_numpy(), dtype=np.float32)  # shape (1000,)
    
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        #data = data / 255.0  # Normalize to [0, 1]
    
    train, test = data[0:train_size], data[train_size:len(data)]
    X_train, y_train = create_sequences(train, sequence_len)
    X_test, y_test = create_sequences(test, sequence_len)    
    return X_train, y_train, X_test, y_test

