import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_and_visualize_correlation(file_path):
    """
    Loads an arbitrary dataset, filters for numerical features,
    calculates the correlation matrix, and displays a heatmap.
    """
    # ----------------------------------------------------------------
    # Step 1: Load the Dataset
    # ----------------------------------------------------------------
    print(f"Loading dataset from: {file_path}")
    try:
        # Handles both CSV and Excel files based on extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please use a .csv or .xlsx file.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # ----------------------------------------------------------------
    # Step 2: Filter for Numerical Features
    # ----------------------------------------------------------------
    # Correlation only works on numbers. This filters out text/categorical columns.
    numerical_df = df.select_dtypes(include=[np.number])
    
    if numerical_df.empty:
        print("Error: No numerical columns found in this dataset to analyze.")
        return
        
    print(f"Analyzing correlation for {numerical_df.shape[1]} numerical features...")

    # ----------------------------------------------------------------
    # Step 3: Calculate the Correlation Matrix
    # ----------------------------------------------------------------
    # Computes standard Pearson correlation coefficients (-1 to +1)
    corr_matrix = numerical_df.corr()

    # ----------------------------------------------------------------
    # Step 4: Visualize the Result 
    # ----------------------------------------------------------------
    # Dynamically adjust the figure size based on how many features exist
    num_features = len(corr_matrix.columns)
    fig_size = max(8, num_features * 0.8)
    
    plt.figure(figsize=(fig_size, fig_size * 0.75))
    
    # Create a mask to hide the upper triangle (removes redundant visual noise)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Generate the heatmap using Seaborn
    sns.heatmap(
        corr_matrix, 
        mask=mask,                  # Hides the upper duplicate half
        cmap='coolwarm',            # Red = Positive correlation, Blue = Negative correlation
        vmax=1.0, vmin=-1.0,        # Sets the strict scale boundaries
        center=0,                   # Pure white means exactly 0.0 (no correlation)
        annot=True,                 # Prints the actual correlation numbers inside squares
        fmt=".2f",                  # Limits numbers to 2 decimal places
        linewidths=0.5,             # Adds clean borders between squares
        cbar_kws={"shrink": 0.8}    # Scales down the color legend bar
    )
    
    plt.title("Dataset Correlation Analysis Heatmap", fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')  # Tilts text so long feature names don't overlap
    plt.yticks(rotation=0)
    plt.tight_layout()          # Prevents margins from clipping labels
    
    # save the correlation matrix
    plt.savefig('results/correlation_heatmap.png', dpi=300, bbox_inches='tight')

# ----------------------------------------------------------------

analyze_and_visualize_correlation('data/space_trajectories_1.csv')