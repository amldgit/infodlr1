import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import a_mert_ffnn_keras as fn
import a_mert_ffnn_keras_cv as fn_cv
from sklearn.model_selection import TimeSeriesSplit

# Load the .mat file as data frame
__mat_data__ = scipy.io.loadmat('Xtrain.mat')
# Convert to DataFrame, there is only one variable in the .mat file
df = pd.DataFrame(__mat_data__['Xtrain']) 

