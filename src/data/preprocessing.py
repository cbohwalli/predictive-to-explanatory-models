import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def apply_vector_scaling(df, num_objects):
    """
    Scales the dataset by dividing by physical maximum boundaries.
    Preserves the (0,0) center of the universe and the directional signs of the vectors.
    """
   
    scaled_df = df.copy()
    
    scale_factors = {
        'mass_max': 500.0,
        'pos_max': 500.0,
        'vel_max': 5.0
    }
    
    # Loop through every object dynamically to scale its columns
    for idx in range(num_objects):
        # 1. Mass scaling (Maps 0 to 500 -> 0 to 1)
        scaled_df[f'obj{idx}_m'] /= scale_factors['mass_max']
        
        # 2. Position scaling (Maps -500 to 500 -> -1 to 1)
        scaled_df[f'obj{idx}_x'] /= scale_factors['pos_max']
        scaled_df[f'obj{idx}_y'] /= scale_factors['pos_max']
        
        # 3. Velocity scaling (Maps -5 to 5 -> -1 to 1)
        scaled_df[f'obj{idx}_vx'] /= scale_factors['vel_max']
        scaled_df[f'obj{idx}_vy'] /= scale_factors['vel_max']
        
    return scaled_df

def prepare_pytorch_datasets(df_scaled, num_objects=3, batch_size=64, val_split=0.2):
    """
    Splits the data by entire simulations to prevent data leakage,
    and returns separate training and validation DataLoaders.
    """
    # 1. Identify all unique simulation universes
    unique_sims = df_scaled['simulation_id'].unique()
    np.random.seed(42) 
    np.random.shuffle(unique_sims)
    
    # 2. Calculate the split boundary
    val_size = int(len(unique_sims) * val_split)
    val_sim_ids = unique_sims[:val_size]
    train_sim_ids = unique_sims[val_size:]
    
    # Helper sub-node to build features and targets from a list of simulation IDs
    def extract_xy(sim_ids):
        X_list, y_list = [], []
        
        # Filter the dataframe for just these simulations
        subset_df = df_scaled[df_scaled['simulation_id'].isin(sim_ids)]
        simulations = subset_df.groupby('simulation_id')
        
        for sim_id, group in simulations:
            group = group.sort_values('frame')
            
            # Dynamic Feature Mapping
            feature_cols = []
            for i in range(num_objects):
                feature_cols.extend([f'obj{i}_m', f'obj{i}_x', f'obj{i}_y', f'obj{i}_vx', f'obj{i}_vy'])
                
            target_cols = []
            for i in range(num_objects):
                target_cols.extend([f'obj{i}_x', f'obj{i}_y', f'obj{i}_vx', f'obj{i}_vy'])
                
            features = group[feature_cols].values
            targets = group[target_cols].values
            
            # X = frame T, Y = frame T+1
            X_list.append(features[:-1])
            y_list.append(targets[1:])
            
        return np.vstack(X_list), np.vstack(y_list)

    X_train, y_train = extract_xy(train_sim_ids)
    X_val, y_val = extract_xy(val_sim_ids)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    # Shuffle the training data to break sequential bias, leave validation un-shuffled
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset Split System:")
    print(f" └── Training Set:   {len(train_sim_ids)} simulations ({X_train.shape[0]} total frames)")
    print(f" └── Validation Set: {len(val_sim_ids)} simulations ({X_val.shape[0]} total frames)")
    
    return train_loader, val_loader