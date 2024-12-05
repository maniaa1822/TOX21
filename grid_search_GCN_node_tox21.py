from GCN_node_tox21 import GCNTox21, train, validate
from sklearn.model_selection import ParameterGrid
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
import json

# Define parameter grids for learning rate scheduler and model architecture
param_grid_1 = {
    'lr': [1e-4, 1e-3],
    'factor': [0.1, 0.5],
    'patience': [2, 5, 10],
    'threshold': [1e-4, 1e-3],
    'min_lr': [1e-6, 1e-5],
}

param_grid_2 = {
    'hidden_dim': [64, 128, 256],
    'num_layers': [3, 4, 5],
    'dropout': [0.2, 0.3, 0.5]
}

# Setup and training
dataset = MoleculeNet(root='data/TOX21', name='TOX21')
train_dataset = dataset[:int(0.8 * len(dataset))]
test_dataset = dataset[int(0.8 * len(dataset)):]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Generate all parameter combinations for both grid_1 and grid_2
grid_1 = ParameterGrid(param_grid_1)
grid_2 = ParameterGrid(param_grid_2)

# Track the best model and performance
best_val_auc = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper function to run grid search for both grids
def run_grid_search(grid, search_name):
    global best_params, best_val_auc
    best_metrics = None

    for params in grid:
        print(f"Testing combination ({search_name}): {params}")
        
        # Determine if we are using grid_1 or grid_2 parameters
        if search_name == "Grid 1":  # These are the learning rate scheduler parameters
            # Initialize model with default parameters for the architecture
            model = GCNTox21(
                num_node_features=dataset[0].num_node_features,
                num_edge_features=dataset[0].num_edge_features,
                hidden_dim=128,  # Default hidden_dim
                num_layers=4,    # Default num_layers
                dropout=0.3      # Default dropout
            ).to(device)
            
            # Initialize optimizer and scheduler
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=params['factor'],
                patience=params['patience'], threshold=params['threshold'],
                min_lr=params['min_lr']
            )
        elif search_name == "Grid 2":  # These are the model architecture parameters
            # Initialize model with parameters from grid_2
            model = GCNTox21(
                num_node_features=dataset[0].num_node_features,
                num_edge_features=dataset[0].num_edge_features,
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            ).to(device)
            
            # Initialize optimizer and scheduler with default settings for this grid
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1,
                patience=5, threshold=1e-4, min_lr=1e-6
            )

        num_epochs = 10
        for epoch in range(num_epochs):
            train_metrics = train(model, train_loader, optimizer, device, epoch)
            val_metrics, _, _ = validate(model, test_loader, device, epoch)

            scheduler.step(val_metrics['loss'])  # Update learning rate scheduler

            # Check if this is the best model
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_params = params
                best_params['search_name'] = search_name
                best_metrics = val_metrics

        print(f"Validation AUC: {val_metrics['auc']:.4f} for parameters: {params}")
    
    if best_metrics is None:  # In case no grid combinations improve the model
        best_metrics = {'auc': 0, 'loss': float('inf')}  # Default metrics

    return best_metrics, best_params

# Run grid searches for both sets of parameters
best_metrics_1, best_params_1 = run_grid_search(grid_1, "Grid 1")
best_metrics_2, best_params_2 = run_grid_search(grid_2, "Grid 2")

# Save best configuration and metrics to JSON
results = {
    'best_config_lr': best_params_1,
    'best_metrics_lr': best_metrics_1,
    'best_config_architecture': best_params_2,
    'best_metrics_architecture': best_metrics_2
}
with open('grid_search/grid_search_GCN_node_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Best configuration and metrics saved to grid_search_results.json.")