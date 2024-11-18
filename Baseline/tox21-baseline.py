"""

This script implements multiple classification models for the Tox21 dataset,
including Random Forest, XGBoost and Gradient Boosting.
The models predict multiple toxicity endpoints using molecular features
derived from SMILES representations of compounds.

Models implemented:
- Random Forest
- XGBoost
- Gradient Boosting

The workflow includes:
1. Data loading and preprocessing
2. Feature generation from SMILES using bag-of-words encoding
3. Train/test splitting and scaling
4. Training multiple models
5. Model comparison and performance evaluation

"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix
)
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
import json
from datetime import datetime
import preprocessing as pp
import matplotlib.pyplot as plt
import seaborn as sns

# Model configurations
MODEL_CONFIGS = {
    'random_forest': {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
    },
    'xgboost': {
        'class': MultiOutputClassifier,
        'base': xgb.XGBClassifier,
        'params': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
    },
    'gradient_boosting': {
        'class': MultiOutputClassifier,
        'base': GradientBoostingClassifier,
        'params': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
    }
}

def load_and_prepare_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Load and prepare the Tox21 dataset for modeling.
    """

    data = pd.read_csv(data_path, index_col=0)
    
    features = ['FW', 'SMILES']
    targets = ['SR-HSE','NR-AR', 'SR-ARE', 'NR-Aromatase', 'NR-ER-LBD', 'NR-AhR', 'SR-MMP',\
       'NR-ER', 'NR-PPAR-gamma', 'SR-p53', 'SR-ATAD5', 'NR-AR-LBD']
    
    X = data[features]
    y = data[targets]
    
    return X, y, targets

def preprocess_data(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the data by handling missing values and creating masks.
    """
    features_types = {'FW': float, 'SMILES': object}
    target_types = {n: float for n in y.columns}
    
    X = X.astype(features_types)
    y = y.astype(target_types)
    
    null_mask = np.array(np.logical_not(y.isnull().values), int)
    y = y.fillna(0.0)
    
    mask_df = pd.DataFrame(
        null_mask, 
        columns=[f'{col}_mask' for col in y.columns], 
        index=y.index
    )
    
    y = pd.concat([y, mask_df], axis=1)
    
    return X, y, mask_df

def create_train_test_split(X: pd.DataFrame, y: pd.DataFrame, mask_df: pd.DataFrame, 
                          test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Split the data into training and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    target_cols = y.columns[:len(mask_df.columns)]
    y_train, mask_train = y_train[target_cols], y_train[mask_df.columns]
    y_test, mask_test = y_test[target_cols], y_test[mask_df.columns]
    
    return X_train, X_test, y_train, y_test, mask_train, mask_test

def generate_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate features from SMILES strings using bag-of-words encoding.
    """
    bow = pp.BagOfWords(X_train['SMILES'].values)
    bow_train = bow.fit()
    bow_test = bow.transform(X_test['SMILES'].values)
    
    bow_train = np.insert(bow_train, 0, X_train['FW'], 1)
    bow_test = np.insert(bow_test, 0, X_test['FW'], 1)
    
    scaler = StandardScaler()
    bow_train = scaler.fit_transform(bow_train)
    bow_test = scaler.transform(bow_test)
    
    return bow_train, bow_test

def initialize_model(model_name: str) -> Any:
    """
    Initialize a model based on the configuration.
    """
    config = MODEL_CONFIGS[model_name]
    if 'base' in config:
        base_model = config['base'](**config['params'])
        return config['class'](base_model)
    else:
        return config['class'](**config['params'])

def train_model(model: Any, X_train: np.ndarray, y_train: np.ndarray, 
                model_path: Optional[str] = None) -> Any:
    """
    Train a model and optionally save it.
    """
    model.fit(X_train, y_train)
    
    if model_path:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    return model

def get_predictions_and_scores(model: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions and probability scores from the model.
    """
    if isinstance(model, MultiOutputClassifier):
        y_pred = model.predict(X)
        y_score = np.array([estimator.predict_proba(X)[:, 1] 
                           for estimator in model.estimators_]).T
    else:
        y_pred = model.predict(X)
        y_score = np.array([model.predict_proba(X)[i][:, 1] 
                           for i in range(12)]).T
    return y_pred, y_score

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_score: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate metrics for binary classification.
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['precision'] = precision_score(y_true, y_pred, sample_weight=sample_weights)
    metrics['recall'] = recall_score(y_true, y_pred, sample_weight=sample_weights)
    metrics['f1'] = f1_score(y_true, y_pred, sample_weight=sample_weights)
    metrics['accuracy'] = accuracy_score(y_true, y_pred, sample_weight=sample_weights)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weights)
    
    # ROC and PR curve metrics
    metrics['roc_auc'] = roc_auc_score(y_true, y_score, sample_weight=sample_weights)
    metrics['average_precision'] = average_precision_score(y_true, y_score, sample_weight=sample_weights)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weights)
    metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
    
    return metrics

def evaluate_model_performance(model: Any, X: np.ndarray, y: np.ndarray, 
                             targets: List[str], sample_weights: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Evaluate model performance across all targets.
    """
    # Get predictions and scores
    y_pred, y_score = get_predictions_and_scores(model, X)
    
    # Calculate metrics for each target
    all_metrics = []
    for i, target in enumerate(targets):
        target_metrics = calculate_metrics(
            y[:, i],
            y_pred[:, i],
            y_score[:, i],
            sample_weights[:, i] if sample_weights is not None else None
        )
        all_metrics.append(target_metrics)
    
    # Create DataFrame with results
    metrics_df = pd.DataFrame(all_metrics, index=targets)
    return metrics_df[['precision', 'recall', 'f1', 'accuracy', 'balanced_accuracy', 
                       'roc_auc', 'average_precision']]

def plot_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, 
                          targets: List[str], output_path: str) -> None:
    """
    Plot confusion matrices for all targets.
    """
    n_targets = len(targets)
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, target in enumerate(targets):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - {target}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_precision_recall_curves(y_true: np.ndarray, y_score: np.ndarray, 
                               targets: List[str], output_path: str) -> None:
    """
    Plot precision-recall curves for all targets.
    """
    plt.figure(figsize=(10, 6))
    
    for i, target in enumerate(targets):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        ap = average_precision_score(y_true[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f'{target} (AP={ap:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def main():
    # Configuration
    DATA_PATH = 'data/data.csv'
    OUTPUT_DIR = Path('output')
    MODELS_DIR = OUTPUT_DIR / 'models'
    METRICS_DIR = OUTPUT_DIR / 'metrics'
    PLOTS_DIR = OUTPUT_DIR / 'plots'
    
    # Create output directories
    for dir_path in [MODELS_DIR, METRICS_DIR, PLOTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    X, y, targets = load_and_prepare_data(DATA_PATH)
    X, y, mask_df = preprocess_data(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test, mask_train, mask_test = create_train_test_split(X, y, mask_df)
    
    # Generate features
    X_train_processed, X_test_processed = generate_features(X_train, X_test)
    
    # Train and evaluate all models
    results = {}
    for model_name in MODEL_CONFIGS.keys():
        print(f"\nTraining {model_name}...")
        
        # Initialize and train model
        model = initialize_model(model_name)
        model_path = MODELS_DIR / f'{model_name}.pkl'
        model = train_model(model, X_train_processed, y_train, str(model_path))
        
        # Evaluate on training set
        print(f"\n{model_name} - Training Set Metrics:")
        train_metrics = evaluate_model_performance(
            model, X_train_processed, y_train.values, targets, mask_train.values
        )
        print(train_metrics)
        
        # Evaluate on test set
        print(f"\n{model_name} - Test Set Metrics:")
        test_metrics = evaluate_model_performance(
            model, X_test_processed, y_test.values, targets, mask_test.values
        )
        print(test_metrics)
        
        # Generate predictions for plots
        y_pred_test, y_score_test = get_predictions_and_scores(model, X_test_processed)
        
        # Create plots
        plot_confusion_matrices(
            y_test.values, y_pred_test, targets,
            str(PLOTS_DIR / f'{model_name}_confusion_matrices.png')
        )
        plot_precision_recall_curves(
            y_test.values, y_score_test, targets,
            str(PLOTS_DIR / f'{model_name}_precision_recall_curves.png')
        )
        
        results[model_name] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
    
    # Compare models
    print("\nModel Comparison (Test Set Performance):")
    comparison = pd.DataFrame({
        model_name: results[model_name]['test_metrics'].mean() 
        for model_name in results.keys()
    })
    print("\nAverage metrics across all targets:")
    print(comparison)
    
    # Save model comparison
    comparison.to_csv(METRICS_DIR / 'model_comparison.csv')

if __name__ == "__main__":
    main()