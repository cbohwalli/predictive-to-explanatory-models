import numpy as np
import pandas as pd

def generate_space_trajectories(num_simulations=100, timesteps_per_sim=200, dt=0.1, num_objects=3):
    """
    Generates a dataset of 2D celestial trajectories for an arbitrary number of objects.
    
    Units are normalized: G = 1.
    Mass range: 1 to 500
    Position range: -500 to 500
    Velocity range: -5 to 5
    """
    G = 1.0
    dataset_rows = []

    for sim_id in range(num_simulations):
        # --- 1. RANDOM INITIALIZATION ---
        # Initialize arrays dynamically based on num_objects
        masses = np.random.uniform(1.0, 500.0, size=num_objects)
        positions = np.random.uniform(-500.0, 500.0, size=(num_objects, 2))
        velocities = np.random.uniform(-5.0, 5.0, size=(num_objects, 2))

        # --- 2. THE PHYSICS SIMULATION LOOP ---
        for step in range(timesteps_per_sim):
            
            # Metadata anchor for the frame
            row = {
                'simulation_id': sim_id,
                'frame': step
            }
            
            # Dynamically generate columns for every single object
            for idx in range(num_objects):
                row.update({
                    f'obj{idx}_m': masses[idx],
                    f'obj{idx}_x': positions[idx, 0],
                    f'obj{idx}_y': positions[idx, 1],
                    f'obj{idx}_vx': velocities[idx, 0],
                    f'obj{idx}_vy': velocities[idx, 1]
                })
                
            dataset_rows.append(row)

            # Calculate Net Force Vectors acting on each object
            accelerations = np.zeros((num_objects, 2))
            
            for i in range(num_objects):
                total_force = np.zeros(2)
                for j in range(num_objects):
                    if i == j:
                        continue
                    
                    # Vector pointing from object i to object j
                    r_vector = positions[j] - positions[i]
                    distance = np.linalg.norm(r_vector)
                    
                    # Softening factor to prevent infinite force during close collisions
                    if distance < 5.0: 
                        distance = 5.0
                        
                    # Gravitational Force magnitude: F = (G * m1 * m2) / r^2
                    force_magnitude = (G * masses[i] * masses[j]) / (distance ** 2)
                    
                    # Direction of force vector
                    direction = r_vector / np.linalg.norm(r_vector)
                    
                    # Add to total force vector accumulator
                    total_force += force_magnitude * direction
                
                # a = F / m
                accelerations[i] = total_force / masses[i]

            # --- 3. UPDATE VECTORS FOR THE NEXT FRAME ---
            positions = positions + velocities * dt + 0.5 * accelerations * (dt ** 2)
            velocities = velocities + accelerations * dt

    # Convert the list of dictionaries into a clean structured DataFrame
    return pd.DataFrame(dataset_rows)

# Execute the generator
df_trajectories = generate_space_trajectories(num_simulations=5, timesteps_per_sim=1000, dt=0.1, num_objects=3)

# Inspect the dataset structure
print(f"Dataset generated successfully! Shape: {df_trajectories.shape}")
print("\nFirst 2 frames of Simulation 0:")
print(df_trajectories.head(2).to_string(index=False))

# Save the Dataset
file_path = 'data/space_trajectories.csv'
df_trajectories.to_csv(file_path, index=False)