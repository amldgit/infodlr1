
# %%
import torch
import a_mert_data as dtu

#This is the .mat file as data frame.
X_train_df = dtu.__df__
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
# Need to split the data into train and test sets. The training data will be split again into k-numbered of folds and
# the item k+1 will be the label. This way I will have the labeled data.
# The parameters to be tuned are: K, number of hidden layers and number of neurons in each layer.
# we will use pytorch to implement the neural network.
# This is a regression problem, so I will use MSE as the loss function and last layer will be only one neuron.