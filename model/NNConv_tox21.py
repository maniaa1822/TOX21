import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool, EdgeConv,NNConv
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
import numpy as np
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

class GCNTox21NNConv(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=64, num_heads=2, num_layers=3, dropout=0.2):
  
        super(GCNTox21NNConv, self).__init__()
        
        # Edge embedding layer
        self.edge_embedding = torch.nn.Linear(num_edge_features, hidden_dim // 2)
        
        # Node embedding layer
        self.node_embedding = torch.nn.Linear(num_node_features, hidden_dim)
        
        # Dynamically create NNConv layers and MLPs
        self.mlps = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        input_dim = hidden_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim // 2
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim // 2, hidden_dim * num_heads),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim * num_heads, input_dim * out_dim)
            )
            self.mlps.append(mlp)
            self.convs.append(NNConv(input_dim, out_dim, mlp, aggr='mean'))
            self.bns.append(torch.nn.BatchNorm1d(out_dim))
            input_dim = out_dim  # Adjust for next layer

        # Fully connected output layer
        self.fc = torch.nn.Linear(hidden_dim // 2, 12)  # Update to 12 tasks
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        # Process edge features
        edge_attr = edge_attr.float()
        edge_embedding = self.edge_embedding(edge_attr)
        
        # Embed node features
        x = self.node_embedding(x)
        
        # Apply NNConv layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index, edge_embedding)
            x = F.relu(bn(x))
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Final classification
        x = self.fc(x)
        x = torch.sigmoid(x)
        
        return x

    
def calculate_metrics(y_true, y_pred, mask):
    """Calculate various performance metrics."""
    # Convert predictions to binary (0/1)
    y_pred_binary = (y_pred > 0.5).float()
    
    # Apply mask to get valid predictions
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    y_pred_binary = y_pred_binary[mask]
    
    # Calculate metrics
    correct = (y_pred_binary == y_true).float().sum()
    total = mask.float().sum()
    accuracy = correct / total
    
    # Calculate precision, recall, f1
    true_positives = ((y_pred_binary == 1) & (y_true == 1)).float().sum()
    predicted_positives = (y_pred_binary == 1).float().sum()
    actual_positives = (y_true == 1).float().sum()
    
    precision = true_positives / (predicted_positives + 1e-10)
    recall = true_positives / (actual_positives + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Calculate ROC-AUC
    try:
        auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    except:
        auc = float('nan')
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'auc': auc
    }

# Create unique run name with timestamp
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = f'runs/NNConv_{current_time}'
writer = SummaryWriter(log_dir)

def train(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    all_metrics = []
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        data.x = data.x.float()
        data.edge_index = data.edge_index.long()
        if data.edge_weight is not None:
            data.edge_weight = data.edge_weight.float()
        optimizer.zero_grad()
        
        # Get model predictions
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        
        # Get all 12 labels
        target = data.y[:, :12].float()
        
        # Calculate BCE loss only on non-nan targets
        mask = ~torch.isnan(target)
        loss = F.binary_cross_entropy(out[mask], target[mask])
        
        # Calculate metrics
        batch_metrics = calculate_metrics(target, out, mask)
        all_metrics.append(batch_metrics)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        
        # Log batch metrics
        if batch_idx % 10 == 0:  # Log every 10 batches
            writer.add_scalar('Batch/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
    
    # Calculate and log epoch metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics]) 
        for key in all_metrics[0].keys()
    }
    avg_metrics['loss'] = total_loss / len(train_loader.dataset)
    
    # Log all metrics
    for metric_name, metric_value in avg_metrics.items():
        writer.add_scalar(f'Train/{metric_name}', metric_value, epoch)
    
    return avg_metrics

@torch.no_grad()
def validate(model, val_loader, device, epoch):
    model.eval()
    total_loss = 0
    all_metrics = []
    all_y_true = []
    all_y_pred = []
    
    for data in val_loader:
        data = data.to(device)
        data.x = data.x.float()
        data.edge_index = data.edge_index.long()
        
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        target = data.y[:, :12].float()
        
        mask = ~torch.isnan(target)
        loss = F.binary_cross_entropy(out[mask], target[mask])
        
        batch_metrics = calculate_metrics(target, out, mask)
        all_metrics.append(batch_metrics)
        total_loss += loss.item() * data.num_graphs
        
        # Collect predictions and true labels
        all_y_true.append(target[mask].cpu())
        all_y_pred.append(out[mask].cpu())
    
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    avg_metrics['loss'] = total_loss / len(val_loader.dataset)
    
    # Log validation metrics
    for metric_name, metric_value in avg_metrics.items():
        writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
    
    # Aggregate all true and predicted values
    all_y_true = torch.cat(all_y_true, dim=0).numpy()
    all_y_pred = torch.cat(all_y_pred, dim=0).numpy()
    
    return avg_metrics, all_y_true, all_y_pred


if __name__ == '__main__':
    # Load best hyperparameters from JSON
    with open('grid_search/grid_search_NNConv_results.json', 'r') as f:
        best_params = json.load(f)
    
    # Combine architecture and learning parameters
    model_params = best_params['best_config_architecture']
    lr_params = best_params['best_config_lr']
    
    # Setup and training
    dataset = MoleculeNet(root='data/TOX21', name='TOX21')
    # Split dataset into train and test 0.8
    train_dataset = dataset[:int(0.8 * len(dataset))]
    test_dataset = dataset[int(0.8 * len(dataset)):]

    # Create data loaders

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
                        

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNTox21NNConv(
        dataset[0].num_node_features, 
        dataset[0].num_edge_features,
        hidden_dim=model_params['hidden_dim'],
        num_heads=model_params['num_heads'],
        num_layers=model_params['num_layers'],
        dropout=model_params['dropout']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_params['lr'])
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=lr_params['factor'],
        patience=lr_params['patience'],
        verbose=True,
        threshold=lr_params['threshold'],
        min_lr=lr_params['min_lr']
    )

    # Save model summary
    model_summary_path = f'{log_dir}/model_summary.txt'
    with open(model_summary_path, 'w') as f:
        f.write(str(model))

    # Print the 12 toxicity tasks we're predicting along with their descriptions
    print("Predicting the following toxicity tasks:")
    task_names = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]
    task_descriptions = [
        'Androgen Receptor',
        'Androgen Receptor Ligand Binding Domain',
        'Aryl Hydrocarbon Receptor',
        'Aromatase',
        'Estrogen Receptor',
        'Estrogen Receptor Ligand Binding Domain',
        'Peroxisome Proliferator-Activated Receptor Gamma',
        'Antioxidant Response Element',
        'ATAD5',
        'Heat Shock Factor Response Element',
        'MMP',
        'p53'
    ]

    for i, (task, description) in enumerate(zip(task_names, task_descriptions)):
        print(f"{i+1}. {task}: {description}")

    # Training loop
    num_epochs = 100
    best_val_auc = 0
    final_y_true, final_y_pred = None, None  # To store the last epoch's predictions

    for epoch in range(num_epochs):
        train_metrics = train(model, train_loader, optimizer, device, epoch)
        val_metrics, y_true, y_pred = validate(model, test_loader, device, epoch)
        
         # Update the learning rate scheduler based on validation loss
        scheduler.step(val_metrics['loss'])
        
        # Store the predictions for the last epoch
        if epoch == num_epochs - 1:
            final_y_true, final_y_pred = y_true, y_pred

        print(f'Epoch {epoch:03d}:')
        print('Training metrics:')
        for k, v in train_metrics.items():
            print(f'  {k}: {v:.4f}')
        print('Validation metrics:')
        for k, v in val_metrics.items():
            print(f'  {k}: {v:.4f}')
        
        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)
        print(f'Learning Rate: {current_lr}')

     # Compute the precision-recall curve
    precision, recall, thresholds = precision_recall_curve(final_y_true.ravel(), final_y_pred.ravel())
    thresholds = np.append(thresholds, 1.0)  # Extend thresholds to match precision/recall length

    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label="PR Curve")
    for i, t in enumerate(thresholds[::len(thresholds)//10]):  # Annotate fewer points for clarity
        plt.text(recall[i], precision[i], f'{t:.2f}', fontsize=8, alpha=0.7)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Final Epoch)')
    plt.legend()
    plt.savefig(f'precision_rcall_curv/NNConv_tox21.png')
    plt.show()

    # Find and print the threshold for a target recall
    target_recall = 0.8
    idx = np.argmin(np.abs(recall - target_recall))
    print(f"Threshold for recall ~{target_recall}: {thresholds[idx]} with precision: {precision[idx]}")

    # Add ROC curve to tensorboard
    fpr, tpr, _ = roc_curve(final_y_true.ravel(), final_y_pred.ravel())
    roc_auc = auc(fpr, tpr)

    # Create figure for ROC curve
    fig_roc = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Add plot to tensorboard
    writer.add_figure('ROC Curve', fig_roc)
    
    # Save ROC curve as image
    plt.savefig(f'{log_dir}/roc_curve.png')
    plt.close()

    # Add final ROC-AUC score to tensorboard
    writer.add_scalar('Final/ROC_AUC', roc_auc)
    
    writer.close()


