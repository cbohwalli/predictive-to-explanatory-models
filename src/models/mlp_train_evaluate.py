import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def train_evaluate_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    # Set device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define the Adam Optimizer
    # lr (learning rate) controls how large of a step the optimizer takes
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Define Loss Function
    criterion = nn.MSELoss()
    
    print(f"Starting training on device: {device}\n" + "-"*40)
    
    for epoch in range(epochs):
        # --- 1. TRAINING PHASE ---
        model.train() # Set model to training mode
        running_train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            # Move data to the active hardware (CPU or GPU)
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Step 1: Clear out gradients from the previous step
            optimizer.zero_grad()
            
            # Step 2: Forward propagation (make predictions)
            predictions = model(batch_X)
            
            # Step 3: Calculate the error gap
            loss = criterion(predictions, batch_y)
            
            # Step 4: Backward propagation (calculate slopes/gradients)
            loss.backward()
            
            # Step 5: Update the network weights using Adam mechanics
            optimizer.step()
            
            # Accumulate the loss for reporting
            running_train_loss += loss.item() * batch_X.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # --- 2. EVALUATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                running_val_loss += loss.item() * batch_X.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        # 1. Convert MSE to RMSE (brings error back to the scaled data unit)
        epoch_val_rmse = np.sqrt(epoch_val_loss)
        
        # 2. Divide by the total universe width (2.0) this locks metric into a clean 0 to 1+ range
        spatial_error_score = epoch_val_rmse / 2.0
        
        # Convert to an explicit percentage for easy reading
        spatial_error_percentage = spatial_error_score * 100
        
        print(f"Epoch {epoch+1:02d} | Train MSE: {epoch_train_loss:.6f} | Val Spatial Error: {spatial_error_percentage:.2f}%")
        
    print("-"*40 + "\nTraining Complete!")
    return model