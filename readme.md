# TOX21 Graph Neural Network Project

## Project Goal
The goal of this project is to develop and evaluate various Graph Neural Network (GNN) models for predicting the toxicity of chemical compounds using the TOX21 dataset. The TOX21 dataset contains information about the biological activities of chemical compounds, which can be represented as graphs where nodes are atoms and edges are bonds.

## Why Graph Neural Networks?
Graph Neural Networks (GNNs) are particularly well-suited for this task because chemical compounds are naturally represented as graphs. In these graphs:
- **Nodes** represent atoms.
- **Edges** represent chemical bonds between atoms.

GNNs can effectively capture the complex relationships and interactions between atoms in a molecule. By leveraging the graph structure, GNNs can:
1. **Aggregate Information**: GNNs aggregate information from neighboring nodes, allowing them to learn the local structure and properties of each atom.
2. **Learn Edge Features**: GNNs can incorporate edge features (e.g., bond types) to better understand the interactions between atoms.
3. **Global Context**: Through multiple layers of graph convolutions, GNNs can capture both local and global structural information, which is crucial for accurately predicting molecular properties.

## Strategies
We employ several GNN architectures to predict the toxicity tasks in the TOX21 dataset:
1. **GCN (Graph Convolutional Network)**: Utilizes graph convolutional layers to aggregate information from neighboring nodes.
2. **GAT (Graph Attention Network)**: Employs attention mechanisms to weigh the importance of neighboring nodes.
3. **NNConv (Neural Network Convolution)**: Uses neural networks to learn edge-specific filters for convolution.
4. **EdgeConv (Edge Convolution)**: Applies convolution operations on edges to capture relationships between nodes.

Each model is trained to predict 12 different toxicity tasks, and we evaluate their performance using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.

