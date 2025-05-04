import numpy as np
import torch
import torch.nn as nn
import mert_data as dt

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size=32, hidden_layers=1):
        super(FNN, self).__init__()        
        
        layers = []
        if hidden_layers == 1:
            layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),      
            nn.Linear(hidden_size, 1)]
        elif hidden_layers == 2:
            layers =[
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),            
            nn.Linear(hidden_size//2, 1)]
        elif hidden_layers > 2:
            layers = []
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            
            layers.append(nn.Linear(hidden_size, hidden_size//2))
            layers.append(nn.ReLU())
            current_size = hidden_size//2
            
            for _ in range(hidden_layers - 2):
                layers.append(nn.Linear(current_size, current_size))
                layers.append(nn.ReLU())
            
            layers.append(nn.Linear(current_size, 1))                
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
    path_best = 'fnn_best_model.pt'
    train_losses = []
    test_losses = []
    best_loss = np.inf
    best_epoch = 0
    final_test_loss = np.inf
    
    def perform_training_no_batch(self,loss_fn, optimizer, epochs=100, train_size=800, sequence_len=5, normalize=True):    
        X_train, y_train, X_test, y_test = dt.split_data(train_size=train_size, sequence_len=sequence_len, normalize=normalize) 
           
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train) # shape (N, window_size)
        y_train = torch.tensor(y_train)
        X_test = torch.tensor(X_test)
        y_test = torch.tensor(y_test)
        
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            output = self(X_train)
            loss = loss_fn(output, y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()
                    
            # Store the training loss
            self.train_losses.append(loss.item())
            
            # Test the model for epoch
            self.eval()
            with torch.no_grad():
                test_output = self(X_test)
                test_loss = loss_fn(test_output, y_test.unsqueeze(1))
                self.test_losses.append(test_loss.item())                
            
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.best_epoch = epoch
                # Save the model state
                torch.save(self.state_dict(), self.path_best)
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
                
        #load the best model
        self.load_state_dict(torch.load('fnn_best_model.pt'))
        self.eval()
        with torch.no_grad():
            predictions = self(X_test)
            self.final_test_loss = loss_fn(predictions, y_test.unsqueeze(1))
            print(f"Final Test MSE: {self.final_test_loss.item():.4f}")

        # Convert tensors to numpy for plotting
        y_test_np = y_test.numpy()        
        predictions_np = predictions.numpy() 
        return self.final_test_loss, y_test_np, predictions_np
            