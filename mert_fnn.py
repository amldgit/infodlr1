import numpy as np
import torch
import torch.nn as nn
import mert_data as dt

def predict(model, X, actual_labels) -> tuple:    
    model.eval()    
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32)
        predictions = model(X)
        train_score = model.loss_fn(predictions, torch.tensor(actual_labels).unsqueeze(1))
        return train_score, predictions.numpy()
        
def recursive_prediction(model, initial_data, num_steps=200):      
    model.eval()  
    preds = []
    input_seq = torch.tensor(initial_data, dtype=torch.float32).unsqueeze(0)  # shape (1, window_size)

    with torch.no_grad():
        for _ in range(num_steps):
            next_pred = model(input_seq)             # shape (1, 1)
            preds.append(next_pred.item())           # store scalar value

            # Update input sequence: drop oldest, append new prediction
            input_seq = torch.cat([input_seq[:, 1:], next_pred], dim=1)  # still shape (1, window_size)

        return np.array(preds)  # convert to numpy array

class FNN(nn.Module):
    path_best = 'fnn_best_model.pt'    
    results = {}
    data_set = []
        
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.reset_results()
    
    def forward(self, x):
        return self.layers(x)        
    
    def reset_results(self):
        self.data_set = []
        self.results = {"train_losses": [], "test_losses": [], "best_loss": np.inf, "best_epoch": 0, "predictions":[], "actual":[] ,"test_score": np.inf, "train_score": np.inf}        
    
    def get_descaled_results(self, copy=True):
        """Returns the results of the training and test losses, predictions and actual values scaled to the original range.
        Args:
            copy (bool, optional): If True, returns a copy of the results. Defaults to True.
        Returns:
            dict: A dictionary containing the descaled results.
        """
        
        if copy:
            #copy the results
            results = self.results.copy()
        else:
            results = self.results
            
        keys_to_descale = ["train_losses", "test_losses","predictions", "actual", "test_score","train_score"]
        for key in keys_to_descale:
            if key in results:
                ds = results[key]                
                results[key] = dt.descale(ds)
        return results
    
    def perform_training(self,epochs=100, sequence_len=5,batch_size=32, shuffle=True):
        """_summary_

        Args:
            epochs (int, optional): Epoch max. limit. Defaults to 100.
            train_size (int, optional): Number of samples in the training dataset. If -1, all data will be used for training. Defaults to -1.
            sequence_len (int, optional): Length of 'memory'. This is the feature size and tells the model how many data points it should look back. Defaults to 5.
            batch_size (int, optional): Number of input batches for optimization. If <= 0, training will run in one single batch. Defaults to 32.
            shuffle (bool, optional): Shuffles the training data. Defaults to True.
        """
        X_train, y_train= dt.preprocess(sequence_len=sequence_len, scale=True)
        self.data_set = [X_train, y_train] #save the dataset for analysis     
        dataset = dt.laser_dataset(X_train, y_train)         
        
        if batch_size <= 0:
            batch_size = len(dataset) # Train the entire dataset in one batch
            
        # Training loop
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            
            # Training step
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.loss_fn(outputs, targets.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Calculate average training loss and strore it
            train_loss = train_loss / len(dataset)                
            self.results["train_losses"].append(train_loss)
            
            # Validation step.            
            # Generate test data from own predictions
            test_d = recursive_prediction(self, X_train[-1], num_steps=200)
            assert len(test_d) == 200, "Test dataset must have 200 samples"
            test_d = test_d.reshape(-1, 1) #reshape to 2D array
            #split the data into sequences and labels
            X_test, y_test = dt.create_sequences(test_d, sequence_len)
            test_loss,_ = predict(self, X_test, y_test)
            self.results["test_losses"].append(test_loss)
            
            if test_loss < self.results["best_loss"]:
                self.results["best_loss"] = test_loss
                self.results["best_epoch"] = epoch
                save_path = 'fnn_best_model.pt'
                torch.save(self.state_dict(), save_path)                               
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Training Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}')                
        
        
        # Generate test data from own predictions        
        test_d = recursive_prediction(self, X_train[-1], num_steps=200)        
        test_d = test_d.reshape(-1, 1) #reshape to 2D array        
        X_test, y_test = dt.create_sequences(test_d, sequence_len)
        
        # Load the best model and evaluate
        self.load_state_dict(torch.load(self.path_best))
        train_score, predictions = predict(self, X_test, y_test)
        self.results["train_score"] = train_score.item()
        self.results["predictions"] = predictions
        self.results["actual"] = dataset.y