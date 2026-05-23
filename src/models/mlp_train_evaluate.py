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
        model.train() # Set model to training mode
        running_loss = 0.0
        
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
            running_loss += loss.item() * batch_X.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Average MSE Loss: {epoch_loss:.6f}")
        
    print("-"*40 + "\nTraining Complete!")
    return model