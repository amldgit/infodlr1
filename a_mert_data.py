import numpy as np
import pandas as pd
import scipy.io
import torch

#The data is in a .mat file. Load it using scipy.io
__mat_data__ = scipy.io.loadmat('Xtrain.mat')
# Convert to DataFrame, there is only one variable in the .mat file
__df__ = pd.DataFrame(__mat_data__['Xtrain']) 

import numpy as np

def __expected_value__(complete_samples:np.array, missing_sample:np.array, idx_missing:int, epsilon=1e-5):
    """This function calculates the expected value at position n for a given sample q, which has missing values.
    It calculates the expected values based on the first n values of complete samples in the X.
    The weighted average of the labels is calculated based on the distance of the first n values of q and the complete samples in X.
    Args:
        known_values = sample_missing_values[:i] (np.array): A 2D array of shape (n_samples, n_features). Array with full data.
        known_values = sample_missing_values[:i] (np.array): Sample with missing values.
        idx_missing (int): The index of the item to be filled.
        epsilon (float, optional): A value to prevent divide by 0. Defaults to 1e-5.

    Returns:
        float: Expected value.
    """
    known_values = complete_samples[:,:idx_missing]
    q_prefix = missing_sample[:idx_missing]
    weights = []
    for xi in known_values:        
        # Calculate the euclidean distance.
        distance = np.linalg.norm(np.array(q_prefix) - np.array(xi)) 
        #Calculate the weight based on the distance. The closer the sample is to the q, the higher the weight.
        weight = 1 / (distance + epsilon)
        weights.append(weight)
    
    y = complete_samples[:,idx_missing]
    weights = np.array(weights)    
    #known_values = np.array(known_values)
    expected = np.sum(weights * y) / np.sum(weights)
    return expected

def _fill_missing_values(complete_sequences:np.array, sample_missing_values:np.array, len_complete:int):
    """This function fills the missing values in the sample with the expected value based on the complete sequences.
    Filling process will start from the index idx_start and will fill all the rest values in the sample.
    Args:
        complete_sequences (np.array): A 2D array of shape (n_samples, n_features). Array with full data.
        sample_missing_values (np.array): Sample with missing values.
        item_of_interest (int): The index of the item to be filled.
    """
    for i in range(len_complete, len(sample_missing_values)):
        # Calculate the expected value based on the first i values of the sample.        
        expected = __expected_value__(complete_sequences, sample_missing_values, i)
        # Set the expected value to the sample.
        sample_missing_values[i] = int(expected)
    
    pass
        
def __generate_sequences(seq_len = 10, n_training_samples = 800):
    #seq_len cannot be less than 2.
    if seq_len < 2:
        raise ValueError("seq_len cannot be less than 2.")
    
    #This is a one dimensional array, that holds the values of the only column in the data frame.
    #Take the n_training_samples.
    data_arr = np.array(__df__[0].values)[0:n_training_samples]
    
    #The data_arr is a single sequence of numbers. We need to split it into K number of sequences.
    #The first K-1 numbers will be the input and the K number will be the label. 
    #If the array cannot be split into K number of sequences, then the last sequence will be discarded.
    #The last sequence will be padded with zeros. We will deal with the padded sample later.
    # Calculate how many complete sequences we can make
    n_complete_sequences = len(data_arr) // seq_len
    # Create the sequences
    sequences = np.array([data_arr[i * seq_len:(i + 1) * seq_len] for i in range(n_complete_sequences)])    
    
    missing_value_count = len(data_arr) % seq_len
    if missing_value_count != 0:
        #Initialize the last sequence with zeros.
        last_sequence = np.zeros(seq_len, dtype=int)
        #Copy the last part of the data_arr to the last sequence.
        last_sequence[:len(data_arr) % seq_len] = data_arr[-(len(data_arr) % seq_len):]

        #The last sequence will be padded with zeros, inluding the label, which is not desired.   
        #Get the index of the first zero starting from the END of the last sequence.
        idx_non_z = len(last_sequence) - 1
        while idx_non_z >= 0 and last_sequence[idx_non_z] == 0:
            idx_non_z -= 1 # reduce the index by 1, if it is still zero.
        n_comp = idx_non_z + 1
        
        #we need to calculate the expected value of the label, based on the first non-zero values of all items in the sequence.
        _fill_missing_values(complete_sequences=sequences, sample_missing_values=last_sequence, len_complete=n_comp)
        sequences = np.append(sequences, [last_sequence], axis=0)
    
    return sequences,missing_value_count 

def create_data_set(seq_len = 10, n_training_samples = 800) -> 'LaserDataSet':
    """This function creates the data set for training and testing.
    It generates the sequences of length seq_len.
    Args:
        seq_len (int): The length of the sequences. Defaults to 10.
        n_training_samples (int): The number of training samples. Defaults to 800.
    """
    #Generate the sequences
    sequences, missing_value_count = __generate_sequences(seq_len, n_training_samples)
    
    #Split the sequences into X and y
    X = sequences[:,:-1]
    y = sequences[:,-1]
    
    return LaserDataSet(X, y)

# test = __generate_data_set(seq_len = 9, n_training_samples = 30)
# print(test)

class LaserDataSet(torch.utils.data.Dataset):  
    
    def __init__(self, X:np.array,y:np.array):
        self.original_X = X
        self.original_y = y

    def __len__(self):
        return len(self.data) - self.K

    def __getitem__(self, idx):
        x = self.original_X[idx]
        y = self.original_y[idx]
        return x, y