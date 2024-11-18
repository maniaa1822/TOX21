import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool, EdgeConv, MessagePassing
from sklearn.metrics import roc_auc_score
import numpy as np
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime


class PairwiseEdgeConv(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super(PairwiseEdgeConv, self).__init__(aggr='mean')
        
        # MLP for processing concatenated features
        # Input: [node_i || node_j || edge_attr]
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels + edge_channels, out_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels * 2, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, edge_channels]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: Source node features [E, in_channels]
        # x_j: Target node features [E, in_channels]
        # edge_attr: Edge features [E, edge_channels]
        
        # Concatenate [source_node || target_node || edge_features]
        out = torch.cat([x_i, x_j, edge_attr], dim=1)
        
        # Apply MLP to concatenated features
        return self.mlp(out)
    

class GCNTox21(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features):
        super(GCNTox21, self).__init__()
        
        # Feature dimensions
        self.node_hidden = 64
        self.edge_hidden = 16
        
        # Edge embedding layer
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.Linear(num_edge_features, self.edge_hidden),
            torch.nn.ReLU()
        )
        
        # Node embedding layer
        self.node_embedding = torch.nn.Sequential(
            torch.nn.Linear(num_node_features, self.node_hidden),
            torch.nn.ReLU()
        )
        
        # Pairwise Edge Convolution layers with dimensions
        # Layer 1: node_hidden -> 128
        self.conv1 = PairwiseEdgeConv(
            in_channels=self.node_hidden,     # 64 per node
            edge_channels=self.edge_hidden,   # 32 for edge
            out_channels=64
        )
        
        # Layer 2: 128 -> 64
        self.conv2 = PairwiseEdgeConv(
            in_channels=64,
            edge_channels=self.edge_hidden,
            out_channels=32
        )
        
        # Layer 3: 64 -> 32
        self.conv3 = PairwiseEdgeConv(
            in_channels=32,
            edge_channels=self.edge_hidden,
            out_channels=16
        )
        
        # Batch normalization layers
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.bn3 = torch.nn.BatchNorm1d(16)
        
        # Final classification layer
        self.fc = torch.nn.Linear(16, 12)  # Update to 12 tasks
        
    def forward(self, x, edge_index, edge_attr, batch):
        # Initial feature dimensions
        # x: [num_nodes, num_node_features]
        # edge_attr: [num_edges, num_edge_features]
        
        # Process edge features
        edge_attr = edge_attr.float()
        edge_embedding = self.edge_embedding(edge_attr)
        # edge_embedding: [num_edges, edge_hidden]
        
        # Embed node features
        x = self.node_embedding(x)
        # x: [num_nodes, node_hidden]
        
        # Apply convolutions with edge features
        x = self.conv1(x, edge_index, edge_embedding)
        x = F.relu(self.bn1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        # x: [num_nodes, 128]
        
        x = self.conv2(x, edge_index, edge_embedding)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        # x: [num_nodes, 64]
        
        x = self.conv3(x, edge_index, edge_embedding)
        x = F.relu(self.bn3(x))
        # x: [num_nodes, 32]
        
        # Global pooling
        x = global_mean_pool(x, batch)
        # x: [batch_size, 32]
        
        # Final classification
        x = self.fc(x)
        x = torch.sigmoid(x)
        # x: [batch_size, 5]
        
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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
                       

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNTox21(dataset[0].num_node_features, dataset[0].num_edge_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

