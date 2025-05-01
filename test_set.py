import a_mert_data as dtu
import numpy as np
import torch

def test_calculate_expectation():
   #The missing data handling algortihm is calculating a missing value in the last sample, 
   #based on the existing values. The algorithm measures the distance of the last sample from the existing samples,
   #and calculates the weighted average of the existing values to predict the missing value.
   #See the method __expected_value__ for more details.
   #For testing we will use the first 3 values of the last sample, and the first 3 values of the existing samples
   #and get the expected value of the label for the last sample.
    
    #Create a sample dataset with only one column.
    #The distances will be: [1, 2, 1] and weights will be [1, 0.5, 1]
    complete_samples = np.array([[1,1], [2,2], [1,1]], dtype=int)
    missing_sample = np.array([0,0])
    #The expected value = 1 * 1 + 0.5 * 2 + 1*1 / (1 + 0.5 + 1) = 1.2, but it will be rounded to 1.
    expected_value = dtu.__expected_value__(complete_samples, missing_sample, 1, epsilon=0)
    assert expected_value == 1.2, f"Expected value is {expected_value}, but it should be 1.0"

def test_missing_data_handling_expectation():
    #Create a sample dataset. 
    #This is extracted from the original data, with training data size 30, sequence length (K) of 9.
    #This makes 3 samples of 9 numbers each, and the last sample has only 3 values, rest is padded with zeros.
    #In total 4 samples, but last one is not complete and needs to be filled with the expected value.
    complete_samples = np.array([[ 86, 141,  95, 41,  22,  21,  32,  72, 138], [111,  48,  23,  19,  27,  59, 129, 129,  58], [ 27,  19,  24,  46, 112, 144,  73,  30,  20]], dtype=int)
    missing_sample = np.array([19, 37, 92,  0,  0,  0,  0,  0,  0])    
    
    #Estimate the 4th value of the last sample, based on the first 3 values of all samples.
    expected_value = dtu.__expected_value__(complete_samples, missing_sample,3)
    expected_value = round(expected_value,3)
    assert expected_value == 37.118, f"Expected value is {expected_value}, but it should be 37.0"
    