import a_mert_data as dtu
import numpy as np
import torch

def test_laser_data_set():
    # Create a sample dataset
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    K = 3

    # Create an instance of the LaserDataSet
    dataset = dtu.LaserDataSet(X, y, K)

    # Check the length of the dataset
    assert len(dataset) == len(X) - K

    # Check the first item
    x_item, y_item = dataset[0]
    assert np.array_equal(x_item.numpy(), X[:K])
    assert y_item == X[K]