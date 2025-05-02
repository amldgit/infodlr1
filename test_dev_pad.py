#ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import mert_data as dtu
import torch.nn as nn
import mert_nn as mnn
from mert_nn import train_loop
from mert_nn import test_loop

n_train = 800
n_test = 200
shuffle_training_set = False
learning_rate = 1e-3
sequence_len = 11 # The length of the sequences, it means (sequence_len - 1) past steps and the last one is the label.
batch_size = 60
mnn.batch_size = batch_size
mnn.n_batch_report = 1

d_train = dtu.create_training_set(seq_len=sequence_len, n_training_samples=n_train, estimate_missing=False)
train_dataloader = torch.utils.data.DataLoader(d_train, batch_size=batch_size, shuffle=shuffle_training_set, num_workers=0)

d_test = dtu.create_test_set(seq_len=sequence_len, n_test_samples=n_test, estimate_missing=False)
test_dataloader = torch.utils.data.DataLoader(d_test, batch_size=batch_size, shuffle=False, num_workers=0)

input_size = sequence_len - 1
model = mnn.MultiLayerPerceptron(input_size=input_size, hidden_size=64)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Convert to while until there is no improvement for x epochs.
epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

