from torch import nn
import torch
import numpy as np

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 32),
        nn.ReLU(),        
        nn.Linear(32, output_size)
    )

    def forward(self, x):
        return self.layers(x)


batch_size = 64
n_batch_report = 100    
batch_results = []
test_results = []

#COPIED FROM: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
#and slighty modified.
def train_loop(dataloader, model, loss_fn, optimizer):
    global batch_results
    
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers    
    model.train()
    for batch, (X, y) in enumerate(dataloader):        
        inputs, targets = X.float(), y.float() #transform to float.
        # Compute the logits and loss
        forward = model(inputs)
        loss = loss_fn(forward, targets)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss, current = loss.item(), batch * batch_size + len(inputs)
        res = {"batch":batch, "loss": loss}
        batch_results.append(res)
        
        if batch % n_batch_report == 0:            
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


#COPIED FROM: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def test_loop(dataloader, model, loss_fn):
    global test_results
    
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
    #correct /= size
    print(f"Avg test loss: {test_loss:>8f} \n")