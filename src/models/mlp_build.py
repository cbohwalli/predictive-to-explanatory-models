import torch.nn as nn

class DynamicMLP(nn.Module):
    """
    A Multi-Layer Perceptron where layer count, layer width, 
    and activation functions are fully parameterized.
    """
    def __init__(self, input_dim, output_dim, hidden_layers=[64], activation='relu'):
        super(DynamicMLP, self).__init__()
        
        # 1. Map string names to PyTorch activation components
        activation_functions = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU()
        }
        
        # Fallback safety default
        act_layer = activation_functions.get(activation.lower(), nn.ReLU())
        
        # 2. Build the structural pipeline of layers
        layers = []
        current_dim = input_dim
        
        # Dynamically construct the hidden structural layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim)) # Linear transformation
            layers.append(act_layer)                          # Non-linear activation
            current_dim = hidden_dim                          # Update input tracking for next layer
            
        # 3. Add the final regression output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        # Bundle the sequential operations into a single modular block
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Executes the forward propagation step."""
        return self.network(x)