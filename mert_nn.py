from torch import nn
import torch
import numpy as np

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, dropout_p=0.2):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
    )

    def forward(self, x):
        return self.layers(x)


batch_size = 64
n_batch_report = 100    
batch_results = []
test_results = []
test_loss_avg = 0
train_loss_first = 0
train_loss_last = 0

def reset_results():
    global batch_results, test_results, test_loss_avg, train_loss_first, train_loss_last
    batch_results = []
    test_results = []

#COPIED FROM: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
#and slighty modified.
def train_loop(dataloader, model, loss_fn, optimizer):
    global batch_results, train_loss_first, train_loss_last
    
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers    
    model.train()
    for batch, (X, y) in enumerate(dataloader):    
        optimizer.zero_grad()    
        # Compute the logits and loss
        forward = model(X)
        loss = loss_fn(forward, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
                
        loss, current = loss.item(), batch * batch_size + len(X)
        res = {"batch":batch, "loss": loss}
        batch_results.append(res)
        
        # if batch % n_batch_report == 0:            
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss_first = batch_results[0]["loss"]
    train_loss_last = batch_results[-1]["loss"]   


#COPIED FROM: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def test_loop(dataloader, model, loss_fn):
    global test_results, test_loss_avg
    
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    batch = 0
    with torch.no_grad():
        for X, y in dataloader:
            batch += 1
            inputs, targets = X.float(), y.float() #transform to float.
            pred = model(inputs)
            loss = loss_fn(pred, targets).item()
            test_loss += loss
            res = {"batch":batch, "loss": loss, "real": np.array(targets), "pred": np.array(pred)}
            test_results.append(res)
            #This needs to be changed for regression.
            #correct += (pred.argmax(1) == targets).type(torch.float).sum().item()

    test_loss /= num_batches
    test_loss_avg = test_loss
    #correct /= size
    #print(f"Test loss. Avg: {test_loss:>8f} \n")