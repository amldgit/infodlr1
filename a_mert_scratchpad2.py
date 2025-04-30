
# %%
import pandas as pd
import scipy.io
import torch

# Load the MATLAB file
mat_data = scipy.io.loadmat('Xtrain.mat')

# Convert to DataFrame - adjust the key if needed
# Usually MATLAB variables are stored with their original name
# If you're not sure about the key, you can print mat_data.keys()
X_train_df = pd.DataFrame(mat_data['Xtrain'])  # Replace 'Xtrain' with actual variable name if different

# %%
# If you want to see the structure of your data
print("DataFrame shape:", X_train_df.shape)
print("\nFirst few rows:")
print(X_train_df.head())

# %%
# It looks like data has only one column. It is expected to train a nerual network with n number of data points, and
# and predict the next data point.
# Lets try to plot the data
import matplotlib.pyplot as plt
plt.plot(X_train_df)
plt.title('X_train Data')
plt.xlabel('Index')
plt.ylabel('Value')
#plt.show() 
# It looks like the data is a sequence of numbers. It was mentioned in the class that it is real world laser data.
# It has some oscillation-like pattern.
# I will split the data into train and test sets. The training data will be split again into k-numbered of folds and
# the item k+1 will be the label. This way I have the labeled data.
# The parameters to be tuned are: K, number of hidden layers and number of neurons in each layer.
# I will use pytorch to implement the neural network.