import numpy as np
import pandas as pd
import scipy.io
import torch

#The data is in a .mat file. Load it using scipy.io
__mat_data__ = scipy.io.loadmat('Xtrain.mat')
# Convert to DataFrame, there is only one variable in the .mat file
__df__ = pd.DataFrame(__mat_data__['Xtrain']) 

import numpy as np

def generate_sequences(data_arr:np.array, row_len = 10, sliding_step=1, discard_last_zero=True) -> tuple:
    
    #12345 12-3 23-4 34-5
    #Create a sliding window of size row_len, with a step of sliding_step. The last row will be padded with zeros.
    #Each row will be a sequence of length row_len, and the last element will be the target.
    sequences = []
    targets = []
    for i in range(0, len(data_arr) - row_len + 1, sliding_step):
        seq = data_arr[i:i + row_len]
        target = data_arr[i + row_len] if i + row_len < len(data_arr) else 0
        sequences.append(seq)
        targets.append(target)

    #If discard_last_zero is True, discard the last row last label is 0. 
    if discard_last_zero:
        sequences = [seq for seq, target in zip(sequences, targets) if target != 0]
        targets = [target for target in targets if target != 0]
    return np.array(sequences), np.array(targets)

class LaserDataSet(torch.utils.data.Dataset):  
    
    def __init__(self, data, seq_length=10):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length].squeeze(-1)
        y = self.data[idx+self.seq_length].unsqueeze(1)
        return x, y 

def create_data_sets(n_training_samples = 800, seq_len = 10, discard_missing_last = True) -> LaserDataSet:
    data_arr = __df__.to_numpy()
    #data_arr = data_arr/255.0  # Normalize to [0, 1]
    training_data = data_arr[:n_training_samples]
    test_data = data_arr[n_training_samples:]
    
    training = LaserDataSet(training_data, seq_length=seq_len)
    test = LaserDataSet(test_data, seq_length=seq_len)
    return training, test