# TOX21 Graph Neural Network Project

## Project Goal
The goal of this project is to develop and evaluate various Graph Neural Network (GNN) models for predicting the toxicity of chemical compounds using the TOX21 dataset. The TOX21 dataset contains information about the biological activities of chemical compounds, which can be represented as graphs where nodes are atoms and edges are bonds.

## TOX 21 Dataset
https://paperswithcode.com/dataset/tox21-1

The Tox21 data set comprises 12,060 training samples and 647 test samples that represent chemical compounds. There are 801 "dense features" that represent chemical descriptors, such as molecular weight, solubility or surface area, and 272,776 "sparse features" that represent chemical substructures (ECFP10, DFS6, DFS8; stored in Matrix Market Format ). Machine learning methods can either use sparse or dense data or combine them. For each sample there are 12 binary labels that represent the outcome (active/inactive) of 12 different toxicological experiments. Note that the label matrix contains many missing values (NAs). The original data source and Tox21 challenge site is https://tripod.nih.gov/tox21/challenge/.

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
1. Data analysis
2. Data preprocessing and cleaning
3. Molecular feature extraction from SMILES notation
4. Dataset stratification and scaling
5. Model training and hyperparameter tuning
6. Comprehensive performance evaluation across multiple metrics

The results provide baseline performance metrics (see table below).


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

### Threshold Selection for Classification

In predictive modeling for toxicity classification, the choice of the threshold value plays a critical role in determining the balance between false positives and false negatives. For the TOX21 dataset, our primary concern was to minimize false negatives, as missing potentially toxic compounds could have severe consequences. However, determining an optimal threshold that achieves a balanced trade-off between precision and recall proved challenging.

#### Analysis of Precision-Recall Trade-off
We evaluated the precision and recall trade-off by plotting the Precision-Recall (PR) curves for each model. The analysis revealed the following:
- **Precision drops as recall improves**: While increasing the recall to 0.8 ensures a higher capture rate of toxic compounds, the precision drops significantly, falling below 0.2. This indicates that a large proportion of the predicted toxic compounds are false positives.
- **No optimal balance**: Despite exploring various threshold values, we were unable to find a point where both precision and recall achieved a satisfactory balance. The inherent trade-off reflects the challenge of working with an imbalanced dataset, where the number of inactive compounds significantly outweighs active ones.

#### Implications of Dataset Imbalance
The TOX21 dataset's imbalance, with far more inactive compounds than active ones, contributes to the skewed precision-recall dynamics. This imbalance affects the model's ability to achieve high recall without sacrificing precision:
- **Recall-focused approach**: Lowering the threshold increases recall, capturing more true positives but also introducing a higher number of false positives.
- **Precision-focused approach**: Raising the threshold improves precision by reducing false positives, but at the cost of missing a substantial number of true positives (reduced recall).

![precision and recall curve of GCN_node model](precision_rcall_curv/GCN_node_tox21.png)

## Grid Search Overview

The table below summarizes the key hyperparameters used for each model in our experiments. These parameters were optimized using grid search to find the best configuration for each model's architecture and task.

### Notes:
- When a parameter is marked with a `-`, it means that the parameter is not applicable to the specific model then it was not included in the grid search optimization for that model.
- the parameters for learning rate and the other parameters are trained separately in order to optimized the computational time.

## Hyperparameters Table

| Models     | Factor | LR      | Min LR   | Patience | Threshold | Dropout | Hidden Dim | Num Heads | Num Layers | Edge Hidden |
|-------------|--------|---------|----------|----------|-----------|---------|------------|-----------|------------|-------------|
| **GAT**     | 0.5    | 0.001   | 1e-06    | 2        | 0.0001    | 0.2     | 64         | 8         | 4          | -           |
| **NNConv**  | 0.5    | 0.001   | 1e-06    | 10       | 0.001     | 0.5     | 256        | 2         | 4          | -           |
| **GCN_node**| 0.1    | 0.001   | 1e-05    | 10       | 0.0001    | 0.2     | 256        | -         | 3          | -           |
| **GCN**     | 0.1    | 0.001   | 1e-05    | 2        | 0.001     | 0.2     | 256        | -         | 3          | 16          |


## Model Performance Results

The following table summarizes the performance results of each model after optimizing the hyperparameters through grid search. This are the performance for the threshold for the best AUC.


### Model Performance Table

| Model       | Accuracy | Precision | Recall | F1     | AUC    | Loss   |
|-------------|----------|-----------|--------|--------|--------|--------|
| **GAT**     | 0.9349   | 0.6621    | 0.2674 | 0.3699 | 0.8328 | 0.2025 |
| **GCN_node**| 0.9369   | 0.6124    | 0.3669 | 0.4536 | 0.8476 | 0.2029 |
| **GCN**     | 0.9336   | 0.7343    | 0.1729 | 0.2686 | 0.8382 | 0.1994 |
| **NNConv**  | 0.9363   | 0.6672    | 0.2977 | 0.4009 | 0.8462 | 0.1983 |

