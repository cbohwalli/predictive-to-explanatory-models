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