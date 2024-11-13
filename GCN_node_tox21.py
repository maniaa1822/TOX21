import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool, EdgeConv
from sklearn.metrics import roc_auc_score
import numpy as np
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime

class GCNTox21(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features):
        super(GCNTox21, self).__init__()
        # Edge embedding layer
        self.edge_embedding = torch.nn.Linear(num_edge_features, 32)
        
        # Node embedding layer
        self.node_embedding = torch.nn.Linear(num_node_features, 64)
        
        # Define MLPs for EdgeConv layers
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(2 * 64, 128),  # Only node features for EdgeConv
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128)
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(2 * 128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64)
        )
        self.mlp3 = torch.nn.Sequential(
            torch.nn.Linear(2 * 64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32)
        )
        
        # Edge convolution layers
        self.conv1 = EdgeConv(self.mlp1, aggr='mean')
        self.conv2 = EdgeConv(self.mlp2, aggr='mean')
        self.conv3 = EdgeConv(self.mlp3, aggr='mean')
        
        # Batch normalization layers
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.fc = torch.nn.Linear(32, 5)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # Process edge and node features
        edge_attr = edge_attr.float()
        edge_embedding = self.edge_embedding(edge_attr)
        
        # Embed node features
        x = self.node_embedding(x)
        
        # Apply convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(self.bn1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = F.relu(self.bn3(x))
        
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
log_dir = f'runs/GCN_tox21_{current_time}'
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
        
        # Get first 5 labels
        target = data.y[:, :5].float()
        
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
    
    for data in val_loader:
        data = data.to(device)
        data.x = data.x.float()
        data.edge_index = data.edge_index.long()
        
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        target = data.y[:, :5].float()
        
        mask = ~torch.isnan(target)
        loss = F.binary_cross_entropy(out[mask], target[mask])
        
        batch_metrics = calculate_metrics(target, out, mask)
        all_metrics.append(batch_metrics)
        total_loss += loss.item() * data.num_graphs
    
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    avg_metrics['loss'] = total_loss / len(val_loader.dataset)
    
    # Log validation metrics
    for metric_name, metric_value in avg_metrics.items():
        writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
    
    return avg_metrics

# Setup and training
dataset = MoleculeNet(root='data/TOX21', name='TOX21')
# Split dataset into train and test 0.8
train_dataset = dataset[:int(0.8 * len(dataset))]
test_dataset = dataset[int(0.8 * len(dataset)):]

# Create data loaders

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
                       

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNTox21(dataset[0].num_node_features, dataset[0].num_edge_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Print the 5 toxicity tasks we're predicting along with their descriptions
print("Predicting the following toxicity tasks:")
task_names = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER'
]
task_descriptions = [
    'Androgen Receptor',
    'Androgen Receptor Ligand Binding Domain',
    'Aryl Hydrocarbon Receptor',
    'Aromatase',
    'Estrogen Receptor'
]

for i, (task, description) in enumerate(zip(task_names, task_descriptions)):
    print(f"{i+1}. {task}: {description}")

# Training loop
num_epochs = 100
best_val_auc = 0
for epoch in range(num_epochs):
    train_metrics = train(model, train_loader, optimizer, device, epoch)
    val_metrics = validate(model, test_loader, device, epoch)
    
    print(f'Epoch {epoch:03d}:')
    print('Training metrics:')
    for k, v in train_metrics.items():
        print(f'  {k}: {v:.4f}')
    print('Validation metrics:')
    for k, v in val_metrics.items():
        print(f'  {k}: {v:.4f}')

# Close writer at the end
writer.close()
