"""
Neural Networks Module
Implements Multi-Layer Perceptron (MLP) for classification and regression
"""

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)


def mlp_classifier(X_train_scaled, y_train, X_test_scaled, y_test, 
                   hidden_layer_sizes=(100, 50), max_iter=500):
    """MLP Classifier"""
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return {
        'model': model,
        'predictions': y_pred,
        'predictions_proba': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'n_iter': model.n_iter_,
        'loss': model.loss_
    }


def mlp_regressor(X_train_scaled, y_train, X_test_scaled, y_test,
                  hidden_layer_sizes=(100, 50), max_iter=500):
    """MLP Regressor"""
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = (mse ** 0.5)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'model': model,
        'predictions': y_pred,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'n_iter': model.n_iter_,
        'loss': model.loss_
    }

