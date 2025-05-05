import numpy as np
import torch
import torch.nn as nn
import mert_data as dt

class FNN(nn.Module):
    path_best = 'fnn_best_model.pt'    
    results = {}
        
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
    
    data_set = []
    
    def perform_training(self,epochs=100, train_size=-1, sequence_len=5,batch_size=32, shuffle=True):
        """_summary_

        Args:
            epochs (int, optional): Epoch max. limit. Defaults to 100.
            train_size (int, optional): Number of samples in the training dataset. If -1, all data will be used for training. Defaults to -1.
            sequence_len (int, optional): Length of 'memory'. This is the feature size and tells the model how many data points it should look back. Defaults to 5.
            batch_size (int, optional): Number of input batches for optimization. If <= 0, training will run in one single batch. Defaults to 32.
            shuffle (bool, optional): Shuffles the training data. Defaults to True.
        """
        X_train, y_train, X_test, y_test = dt.split_data(train_size=train_size, sequence_len=sequence_len, scale=True)
        self.data_set = [X_train, y_train, X_test, y_test] #save the dataset for analysis     
        X_Train = dt.laser_dataset(X_train, y_train)
        y_train = dt.laser_dataset(y_train, y_train)
        X_test = dt.laser_dataset(X_test, y_test)
        y_test = dt.laser_dataset(y_test, y_test)
        
        if batch_size <= 0:
            batch_size = len(X_Train) # Train the entire dataset in one batch
            
        # Training loop
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            
            # Training step
            train_loader = torch.utils.data.DataLoader(X_Train, batch_size=batch_size, shuffle=shuffle)
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.loss_fn(outputs, targets.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Calculate average training loss and strore it
            train_loss = train_loss / len(X_Train)                
            self.results["train_losses"].append(train_loss)
            
            # Validation step. Only if there is a test set.
            test_loss = np.nan
            if len(X_test) > 0:
                self.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in torch.utils.data.DataLoader(X_test, batch_size=batch_size):
                        outputs = self(inputs)
                        test_loss += self.loss_fn(outputs, targets.unsqueeze(1)).item()
                
                test_loss = test_loss / len(X_test)
                self.results["test_losses"].append(test_loss)
                
                if test_loss < self.results["best_loss"]:
                    self.results["best_loss"] = test_loss
                    self.results["best_epoch"] = epoch
                    save_path = 'fnn_best_model.pt'
                    torch.save(self.state_dict(), save_path)                
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Training Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}')
                #print(f'Epoch {epoch}: Training Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}')
                
        if len(X_test) == 0:
            # Save the model if there is no test set
            torch.save(self.state_dict(), self.path_best)
        #load the best model
        # self.load_state_dict(torch.load('fnn_best_model.pt'))
        # self.eval()
        
        # Make predictions
        with torch.no_grad():
            # Generate test score
            # test_loader = torch.utils.data.DataLoader(X_test, batch_size=len(X_test))
            # for inputs, _ in test_loader:
            #     predictions = self(inputs.float())
            
            # test_score = self.loss_fn(predictions, torch.tensor(y_test.y).unsqueeze(1))
            # self.results["test_score"] = test_score.item()
            
            # # Convert tensors to numpy for plotting
            # self.results["actual"] = y_test.y     
            # self.results["predictions"] = predictions.numpy() 
            
            # Generate training score
            # train_loader = torch.utils.data.DataLoader(X_Train, batch_size=len(X_Train))
            # for inputs, _ in train_loader:
            #     t_predictions = self(inputs.float())
            # train_score = self.loss_fn(t_predictions, torch.tensor(y_train.y).unsqueeze(1))
            train_score, predictions = self.predict(X_train, y_train.y, self.path_best)
            self.results["train_score"] = train_score.item()
    
    def predict(self, X, actual_labels, model_path:str) -> np.array:
        """Predicts the output for the given input data.
        Args:
            X (array-like): Input data.
        Returns:
            array: Predicted output.
        """
        self.load_state_dict(torch.load(model_path))
        self.eval()
        
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            predictions = self(X)
            train_score = self.loss_fn(predictions, torch.tensor(actual_labels).unsqueeze(1))
            return train_score, predictions.numpy()