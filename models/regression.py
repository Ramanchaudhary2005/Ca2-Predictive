"""
Regression Models Module
Implements Simple Linear, Multiple Linear, and Polynomial Regression
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def simple_linear_regression(X_train, y_train, X_test, y_test):
    """Simple Linear Regression: Predict Global_Sales from NA_Sales"""
    # Use only NA_Sales
    X_simple_train = X_train[['NA_Sales']].values
    X_simple_test = X_test[['NA_Sales']].values
    
    model = LinearRegression()
    model.fit(X_simple_train, y_train)
    
    y_pred = model.predict(X_simple_test)
    
    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'model': model,
        'predictions': y_pred,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'coefficient': model.coef_[0],
        'intercept': model.intercept_
    }


def multiple_linear_regression(X_train_scaled, y_train, X_test_scaled, y_test):
    """Multiple Linear Regression using all features"""
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'model': model,
        'predictions': y_pred,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'coefficients': model.coef_
    }


def polynomial_regression(X_train, y_train, X_test, y_test, degree=3):
    """Polynomial Regression using Year as feature"""
    # Use only Year
    X_poly_train = X_train[['Year']].values
    X_poly_test = X_test[['Year']].values
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly_train_transformed = poly_features.fit_transform(X_poly_train)
    X_poly_test_transformed = poly_features.transform(X_poly_test)
    
    # Fit polynomial regression
    model = LinearRegression()
    model.fit(X_poly_train_transformed, y_train)
    
    y_pred = model.predict(X_poly_test_transformed)
    
    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'model': model,
        'poly_features': poly_features,
        'predictions': y_pred,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

