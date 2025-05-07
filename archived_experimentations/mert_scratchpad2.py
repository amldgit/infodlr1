
# %%
import torch
import mert_data as dtu
#This is the .mat file as data frame.
X_train_df = dtu.__df__
# %%
# It looks like data has only one column. It is expected to train a nerual network with n number of data points, and
# and predict the next data point.
# Lets try to plot the data
import matplotlib.pyplot as plt
plt.plot(X_train_df)
plt.title('X_train Data')
plt.xlabel('Index')
plt.ylabel('Value')
mean = X_train_df.mean()[0]
std = X_train_df.std()[0]
plt.axhline(y=mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
plt.axhline(y=mean + std, color='g', linestyle=':', label=f'Mean ± Std: {mean:.2f} ± {std:.2f}')
plt.axhline(y=mean - std, color='g', linestyle=':')
plt.legend()
#plt.show() 
# It looks like the data is a sequence of numbers. It was mentioned in the class that it is real world laser data.
# It has some oscillation-like pattern.
# Need to split the data into train and test sets. The training data will be split again into k-numbered of folds and
# the item k+1 will be the label. This way I will have the labeled data.
# The parameters to be tuned are: K, number of hidden layers and number of neurons in each layer.
# we will use pytorch to implement the neural network.
# This is a regression problem, so I will use MSE as the loss function and last layer will be only one neuron.
# %%

# Calculate descriptive statistics
desc_stats = X_train_df.describe()
print("\nDescriptive Statistics:")
print(desc_stats)

# Create a box plot
plt.figure(figsize=(10, 6))
plt.boxplot(X_train_df.values)
plt.title('Box Plot of Data Distribution')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(X_train_df.values, bins=50, density=True)
plt.title('Histogram of Data Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()
# %%
# use scikit standard scaler to scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X_train_df)

plt.figure(figsize=(10, 6))
plt.plot(scaled_data)
plt.title('Scaled Data')
plt.xlabel('Index')
plt.ylabel('Standardized Value')
plt.grid(True)
plt.show()

# Create a box plot for scaled data
plt.figure(figsize=(10, 6))
plt.boxplot(scaled_data)
plt.title('Box Plot of Scaled Data Distribution')
plt.ylabel('Standardized Value')
plt.grid(True)
plt.show()


# %%
