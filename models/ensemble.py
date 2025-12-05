"""
Ensemble Methods Module
Implements Bagging, Boosting (AdaBoost), and Random Forest
"""

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def bagging_classifier(X_train_scaled, y_train, X_test_scaled, y_test,
                       n_estimators=50, max_depth=5):
    """Bagging Classifier"""
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth),
        n_estimators=n_estimators,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return {
        'model': model,
        'predictions': y_pred,
        'predictions_proba': y_pred_proba,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }


def adaboost_classifier(X_train_scaled, y_train, X_test_scaled, y_test,
                        n_estimators=50, max_depth=3):
    """AdaBoost Classifier"""
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth),
        n_estimators=n_estimators,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return {
        'model': model,
        'predictions': y_pred,
        'predictions_proba': y_pred_proba,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }


def random_forest_classifier(X_train_scaled, y_train, X_test_scaled, y_test,
                             n_estimators=100, max_depth=10):
    """Random Forest Classifier"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return {
        'model': model,
        'predictions': y_pred,
        'predictions_proba': y_pred_proba,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'feature_importance': model.feature_importances_
    }

