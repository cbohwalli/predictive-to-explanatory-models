import torch
import pandas as pd

import src.data.preprocessing as prep
import src.models.mlp_build as build
import src.models.mlp_train_evaluate as train_eval

# Todo: add script of generating datasets in the begining of script
# Todo: make the experiment modular

def main():
    # --- GLOBAL HYPERPARAMETERS ---
    NUM_OBJECTS = 3
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 0.001
    HIDDEN_ARCHITECTURE = [128, 64]
    ACTIVATION_FUNCTION = 'relu'

    # --- 1. LOAD DATASET ---
    df_raw_medium = pd.read_csv("data/medium_size.csv")
    df_raw_large = pd.read_csv("data/large_size.csv") 
  
    # --- 2. PREPROCESS DATASET ---

    df_scaled = prep.apply_vector_scaling(df_raw_large, NUM_OBJECTS)
    
    train_loader, val_loader = prep.prepare_pytorch_datasets(
        df_scaled, 
        num_objects=NUM_OBJECTS, 
        batch_size=BATCH_SIZE, 
        val_split=0.2
    )

    # --- 3. BUILD MODEL ---

    # Calculate dimensional shapes based on object count
    # Input: mass, x, y, vx, vy for all objects (3 * 5 = 15)
    # Output: x, y, vx, vy for all objects (3 * 4 = 12)
    input_dim = NUM_OBJECTS * 5
    output_dim = NUM_OBJECTS * 4
    
    model = build.DynamicMLP(
        input_dim=input_dim, 
        output_dim=output_dim, 
        hidden_layers=HIDDEN_ARCHITECTURE, 
        activation=ACTIVATION_FUNCTION
    )

    # --- 4. TRAIN & EVALUATE MODEL ---
    # The training function runs validation evaluation automatically at the end of each epoch
    trained_model = train_eval.train_evaluate_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE
    )
    
    # --- 5. SAVE ARTIFACT ---
    print("\nSaving trained model weights...")
    torch.save(trained_model.state_dict(), "src/models/gravity_mlp_weights.pth")
    print(" └── Weights successfully saved to 'src/models/gravity_mlp_weights.pth'")
    print("\n🎉 Experiment complete!")
    print("="*60)

if __name__ == "__main__":
    main()