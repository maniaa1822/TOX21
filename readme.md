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

## Baseline Methods

To establish performance benchmarks before implementing more complex Graph Neural Networks (GNNs), we evaluated several classical machine learning approaches on the Tox21 dataset. These baseline models provide a foundation for comparing the effectiveness of more sophisticated architectures.

### Implementation

We developed three tree-based ensemble models:
- Random Forest
- XGBoost
- Gradient Boosting

Each model predicts multiple toxicity endpoints using molecular features extracted from SMILES representations. For molecular featurization, we employed a bag-of-words encoding approach, which converts chemical structures into numerical vectors while preserving key molecular information.

### Methodology

Our experimental pipeline consists of:
1. Data preprocessing and cleaning
2. Molecular feature extraction from SMILES notation
3. Dataset stratification and scaling
4. Model training and hyperparameter tuning
5. Comprehensive performance evaluation across multiple metrics

The results provide baseline performance metrics (see table below) against which future GNN implementations can be compared.


### Baseline Performance Comparison

| Metric | Random Forest | XGBoost | Gradient Boosting |
|--------|--------------|---------|------------------|
| Precision | 79.69% | 73.33% | 56.22% |
| Recall | 9.27% | 18.01% | 20.71% |
| F1 Score | 13.91% | 27.31% | 29.10% |
| Accuracy | 92.70% | 93.12% | 92.91% |
| Balanced Accuracy | 54.62% | 58.76% | 59.87% |
| ROC AUC | 81.07% | 79.52% | 79.29% |
| Average Precision | 40.77% | 41.06% | 36.91% |

Key observations:
- Random Forest achieves highest precision and ROC AUC
- Gradient Boosting shows best recall and balanced accuracy
- XGBoost provides best overall balance with highest accuracy and average precision

