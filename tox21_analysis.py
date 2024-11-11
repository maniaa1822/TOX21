#%% import
import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm
from IPython.display import display
import networkx as nx
from torch_geometric.utils import to_networkx
from tabulate import tabulate

#!pip install torch torch-geometric numpy pandas matplotlib seaborn rdkit tqdm networkx ipython
%matplotlib inline

#%% Load dataset
dataset = MoleculeNet(root='data/TOX21', name='TOX21')

dataset.get_summary()

#%% Basic Dataset Statistics
print("\n=== Basic Dataset Statistics ===")
print(f"Total number of molecules: {len(dataset)}")
print(f"Number of node features: {dataset.num_node_features}")
print(f"Number of edge features: {dataset.num_edge_features}")
print(f"Number of tasks: {dataset.num_classes}")

#%% Graph Statistics
print("\n=== Graph Statistics ===")
n_nodes = []
n_edges = []
node_degrees = []

for data in tqdm(dataset, desc="Analyzing graphs"):
    n_nodes.append(data.num_nodes)
    n_edges.append(data.num_edges)
    degrees = torch.bincount(data.edge_index[0])
    node_degrees.extend(degrees.tolist())

print(f"Average number of nodes: {np.mean(n_nodes):.2f} ± {np.std(n_nodes):.2f}")
print(f"Average number of edges: {np.mean(n_edges):.2f} ± {np.std(n_edges):.2f}")
print(f"Average node degree: {np.mean(node_degrees):.2f} ± {np.std(node_degrees):.2f}")

#%% Visualize distributions
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.hist(n_nodes, bins=50)
ax1.set_title('Distribution of Number of Nodes')
ax1.set_xlabel('Number of Nodes')
ax1.set_ylabel('Frequency')

ax2.hist(n_edges, bins=50)
ax2.set_title('Distribution of Number of Edges')
ax2.set_xlabel('Number of Edges')
ax2.set_ylabel('Frequency')

ax3.hist(node_degrees, bins=50)
ax3.set_title('Distribution of Node Degrees')
ax3.set_xlabel('Node Degree')
ax3.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

#%% Feature Analysis
print("\n=== Feature Analysis ===")
# Analyze node features
node_features = torch.cat([data.x for data in dataset], dim=0).float()

print("Node Features Statistics:")
print(f"Feature dimensionality: {node_features.shape[1]}")

# Calculate feature statistics
feature_means = node_features.mean(dim=0)
feature_stds = node_features.std(dim=0)

#%% Plot feature statistics
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.bar(range(len(feature_means)), feature_means)
plt.title('Mean Values of Node Features')
plt.xlabel('Feature Index')
plt.ylabel('Mean Value')

plt.subplot(1, 2, 2)
plt.bar(range(len(feature_stds)), feature_stds)
plt.title('Standard Deviation of Node Features')
plt.xlabel('Feature Index')
plt.ylabel('Standard Deviation')

plt.tight_layout()
plt.show()

#%% Task Correlations
print("\n=== Task Correlations ===")
# All labels
all_labels = torch.stack([data.y for data in dataset]).squeeze().numpy()
# Define task names
task_names = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
    'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
    'SR-HSE', 'SR-MMP', 'SR-p53'
]
# Calculate correlation matrix (ignoring missing values)
correlation_matrix = np.zeros((len(task_names), len(task_names)))

for i in range(len(task_names)):
    for j in range(len(task_names)):
        # Get valid indices (where both tasks have values)
        valid_indices = ~np.isnan(all_labels[:, i]) & ~np.isnan(all_labels[:, j])
        if valid_indices.sum() > 0:
            correlation_matrix[i, j] = np.corrcoef(
                all_labels[valid_indices, i],
                all_labels[valid_indices, j]
            )[0, 1]

#%% Plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, 
            xticklabels=task_names, 
            yticklabels=task_names, 
            annot=True, 
            cmap='coolwarm',
            vmin=-1, 
            vmax=1)
plt.title('Task Correlations')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

#%% Summary Statistics
print("\n=== Summary Statistics ===")
print(f"Dataset size: {len(dataset)} molecules")
print("\nPer-task statistics:")

missing_values = []

for i, task in enumerate(task_names):
    # Count missing values
    missing = np.isnan(all_labels[:, i]).sum()
    missing_values.append(missing)

for i, task in enumerate(task_names):
    valid_labels = all_labels[:, i][~np.isnan(all_labels[:, i])]
    pos_rate = (valid_labels == 1).mean() if len(valid_labels) > 0 else 0
    missing_rate = missing_values[i] / len(dataset)
    
    print(f"\n{task}:")
    print(f"  - Positive rate: {pos_rate:.3f}")
    print(f"  - Missing rate: {missing_rate:.3f}")
    print(f"  - Total valid samples: {len(valid_labels)}")

#%% Additional analysis for all_labels
all_labels = np.array([data.y for data in dataset])
print(f"Shape of all_labels: {all_labels.shape}")

for i in range(all_labels.shape[1]):
    missing = np.isnan(all_labels[:, i]).sum()
    print(f"Missing values in column {i}: {missing}")

#%% Visualize a sample of molecules
print("\n=== Visualize Sample Molecules ===")
sample_molecules = [dataset[i] for i in range(10)]
sample_smiles = [data.smiles for data in sample_molecules]
sample_mols = [Chem.MolFromSmiles(smiles) for smiles in sample_smiles]

img = Draw.MolsToGridImage(sample_mols, molsPerRow=5, subImgSize=(200, 200))
display(img)

#%% Visualize the graph of a sample molecule !!! test
print("\n=== Visualize Graph of a Sample Molecule ===")
sample_data = dataset[1]  # Take the first molecule as an example
G = to_networkx(sample_data, node_attrs=['x'], edge_attrs=['edge_attr'])

# Plot the graph
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)

# Draw node labels
node_labels = {i: f"{i}\n{G.nodes[i]['x']}" for i in G.nodes}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

# Draw edge labels
edge_labels = {(i, j): f"{G.edges[i, j]['edge_attr']}" for i, j in G.edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# Create a legend for node attributes with actual feature names
node_feature_names = ['Atomic Number', 'Chirality', 'Degree', 'Formal Charge', 'Num Hs', 'Hybridization']
node_attr_legend = {i: node_feature_names[i] for i in range(len(node_feature_names))}
node_attr_legend_text = "\n".join([f"{k}: {v}" for k, v in node_attr_legend.items()])

# Create a legend for edge attributes with actual feature names
edge_feature_names = ['Bond Type', 'Bond Stereo']
edge_attr_legend = {i: edge_feature_names[i] for i in range(len(edge_feature_names))}
edge_attr_legend_text = "\n".join([f"{k}: {v}" for k, v in edge_attr_legend.items()])

# Display the legends
plt.gcf().text(0.02, 0.5, f"Node Attributes:\n{node_attr_legend_text}", fontsize=10, verticalalignment='center')
plt.gcf().text(0.98, 0.5, f"Edge Attributes:\n{edge_attr_legend_text}", fontsize=10, verticalalignment='center', horizontalalignment='right')

plt.title('Graph of a Sample Molecule')
plt.show()

# %%
