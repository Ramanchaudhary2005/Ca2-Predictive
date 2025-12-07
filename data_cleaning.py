"""
Data Cleaning and Preprocessing Module
Handles all data loading, cleaning, feature engineering, and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(file_path='video_games_sales.csv'):
    """Load the video games sales dataset"""
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    """Clean the dataset: handle missing values, remove duplicates, standardize"""
    df_clean = df.copy()
    
    # Handle missing values in Year column
    df_clean = df_clean.dropna(subset=['Year'])
    df_clean['Year'] = df_clean['Year'].astype(int)
    
    # Handle missing values in Publisher
    df_clean['Publisher'] = df_clean['Publisher'].fillna('Unknown')
    
    # Standardize text columns
    df_clean['Genre'] = df_clean['Genre'].str.strip()
    df_clean['Platform'] = df_clean['Platform'].str.strip()
    df_clean['Publisher'] = df_clean['Publisher'].str.strip()
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    return df_clean


def engineer_features(df_clean):
    """Create new features for modeling"""
    # Create binary classification target: Hit (>=1M sales) or Flop (<1M sales)
    df_clean['Hit'] = (df_clean['Global_Sales'] >= 1.0).astype(int)
    
    # Age of game (assuming current year is 2020 for this dataset)
    df_clean['Game_Age'] = 2020 - df_clean['Year']
    
    # Total regional sales (sum of all regions)
    df_clean['Total_Regional_Sales'] = (df_clean['NA_Sales'] + df_clean['EU_Sales'] + 
                                        df_clean['JP_Sales'] + df_clean['Other_Sales'])
    
    # Publisher success rate (average global sales for publisher)
    publisher_avg = df_clean.groupby('Publisher')['Global_Sales'].mean().to_dict()
    df_clean['Publisher_Avg_Sales'] = df_clean['Publisher'].map(publisher_avg)
    
    # Genre-Platform combination popularity (average sales for this combo)
    df_clean['Genre_Platform_Combo'] = df_clean['Genre'] + '_' + df_clean['Platform']
    combo_avg = df_clean.groupby('Genre_Platform_Combo')['Global_Sales'].mean().to_dict()
    df_clean['Combo_Avg_Sales'] = df_clean['Genre_Platform_Combo'].map(combo_avg)
    
    # Year trend (normalized year - helps capture market evolution)
    df_clean['Year_Normalized'] = (df_clean['Year'] - df_clean['Year'].min()) / (df_clean['Year'].max() - df_clean['Year'].min())
    
    return df_clean


def encode_categorical(df_clean):
    """Encode categorical variables using Label Encoding"""
    le_genre = LabelEncoder()
    le_platform = LabelEncoder()
    le_publisher = LabelEncoder()
    
    df_clean['Genre_Encoded'] = le_genre.fit_transform(df_clean['Genre'])
    df_clean['Platform_Encoded'] = le_platform.fit_transform(df_clean['Platform'])
    df_clean['Publisher_Encoded'] = le_publisher.fit_transform(df_clean['Publisher'])
    
    return df_clean, le_genre, le_platform, le_publisher


def prepare_data(df_clean):
    """Prepare features and targets for regression and classification"""
    # Regression features and target
    regression_features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Year', 
                           'Genre_Encoded', 'Platform_Encoded', 'Publisher_Encoded',
                           'Game_Age', 'Total_Regional_Sales', 'Publisher_Avg_Sales', 
                           'Combo_Avg_Sales', 'Year_Normalized']
    
    X_reg = df_clean[regression_features]
    y_reg = df_clean['Global_Sales']
    
    # Classification features and target
    classification_features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Year',
                              'Genre_Encoded', 'Platform_Encoded', 'Publisher_Encoded',
                              'Game_Age', 'Total_Regional_Sales', 'Publisher_Avg_Sales',
                              'Combo_Avg_Sales', 'Year_Normalized']
    
    X_clf = df_clean[classification_features]
    y_clf = df_clean['Hit']
    
    return X_reg, y_reg, X_clf, y_clf, regression_features, classification_features


def split_and_scale(X_reg, y_reg, X_clf, y_clf, test_size=0.2, random_state=42):
    """Split data into train/test sets and scale features"""
    # Train-Test Split
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=test_size, random_state=random_state
    )
    
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
    )
    
    # Scaling
    scaler_reg = StandardScaler()
    scaler_clf = StandardScaler()
    
    X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
    X_reg_test_scaled = scaler_reg.transform(X_reg_test)
    
    X_clf_train_scaled = scaler_clf.fit_transform(X_clf_train)
    X_clf_test_scaled = scaler_clf.transform(X_clf_test)
    
    return {
        'X_reg_train': X_reg_train,
        'X_reg_test': X_reg_test,
        'y_reg_train': y_reg_train,
        'y_reg_test': y_reg_test,
        'X_clf_train': X_clf_train,
        'X_clf_test': X_clf_test,
        'y_clf_train': y_clf_train,
        'y_clf_test': y_clf_test,
        'X_reg_train_scaled': X_reg_train_scaled,
        'X_reg_test_scaled': X_reg_test_scaled,
        'X_clf_train_scaled': X_clf_train_scaled,
        'X_clf_test_scaled': X_clf_test_scaled,
        'scaler_reg': scaler_reg,
        'scaler_clf': scaler_clf
    }


def get_correlation_matrix(df_clean):
    """Get correlation matrix for visualization"""
    correlation_matrix = df_clean[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 
                                   'Global_Sales', 'Year', 'Genre_Encoded', 'Platform_Encoded']].corr()
    return correlation_matrix

