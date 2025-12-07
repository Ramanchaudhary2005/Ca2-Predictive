"""
Streamlit Web Application for Predictive Analytics Project
Video Games Sales Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram

# Import custom modules
import data_cleaning as dc
from models import regression, classification, clustering, neural_networks, ensemble

# Page configuration
st.set_page_config(
    page_title="Predictive Analytics Project",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎮 Video Games Sales Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# ----- REPLACED: Sidebar radio navigation with a top navbar -----
pages = [
    "📊 Dataset Overview", "🔧 Data Preprocessing", "📈 Regression Models",
    "🎯 Classification Models", "🔍 Clustering", "🧠 Neural Networks",
    "⚡ Ensemble Methods", "📋 Model Comparison", "🎮 Game Prediction"
]

# ensure session_state key exists
if 'page' not in st.session_state:
    st.session_state['page'] = pages[0]

# render navbar as a row of buttons
nav_cols = st.columns(len(pages))
for i, p in enumerate(pages):
    if nav_cols[i].button(p):
        st.session_state['page'] = p

page = st.session_state['page']
# -----------------------------------------------------------------

# Cache management in sidebar
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
if st.sidebar.button("🔄 Clear Cache & Reload", help="Clear all cached data and models. Use this if you encounter feature mismatch errors."):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("Cache cleared! Please refresh the page.")
    st.rerun()

# Cache version - increment this to force cache refresh when features change
FEATURE_VERSION = "v2.0"  # Updated for 13 features

# Load data (cached)
@st.cache_data
def load_and_preprocess_data(_version):
    """Load and preprocess data"""
    df = dc.load_data('video_games_sales.csv')
    df_clean = dc.clean_data(df)
    df_clean = dc.engineer_features(df_clean)
    df_clean, le_genre, le_platform, le_publisher = dc.encode_categorical(df_clean)
    X_reg, y_reg, X_clf, y_clf, reg_features, clf_features = dc.prepare_data(df_clean)
    data_splits = dc.split_and_scale(X_reg, y_reg, X_clf, y_clf)
    
    return {
        'df': df,
        'df_clean': df_clean,
        'X_reg': X_reg,
        'y_reg': y_reg,
        'X_clf': X_clf,
        'y_clf': y_clf,
        'reg_features': reg_features,
        'clf_features': clf_features,
        'splits': data_splits,
        'le_genre': le_genre,
        'le_platform': le_platform,
        'le_publisher': le_publisher
    }

# Train prediction models (cached)
@st.cache_resource
def train_prediction_models(_splits, _version):
    """Train and cache the best models for prediction"""
    # Train Random Forest for classification (best ensemble model)
    clf_model = ensemble.random_forest_classifier(
        _splits['X_clf_train_scaled'], _splits['y_clf_train'],
        _splits['X_clf_test_scaled'], _splits['y_clf_test'],
        n_estimators=100, max_depth=10
    )
    
    # Train Multiple Linear Regression for sales prediction
    reg_model = regression.multiple_linear_regression(
        _splits['X_reg_train_scaled'], _splits['y_reg_train'],
        _splits['X_reg_test_scaled'], _splits['y_reg_test']
    )
    
    return {
        'clf_model': clf_model['model'],
        'reg_model': reg_model['model'],
        'clf_scaler': _splits['scaler_clf'],
        'reg_scaler': _splits['scaler_reg']
    }

# Get average regional sales for genre/platform combination
@st.cache_data
def get_average_regional_sales(df_clean):
    """Calculate average regional sales by genre and platform"""
    avg_sales = df_clean.groupby(['Genre', 'Platform']).agg({
        'NA_Sales': 'mean',
        'EU_Sales': 'mean',
        'JP_Sales': 'mean',
        'Other_Sales': 'mean'
    }).reset_index()
    return avg_sales

# Get publisher statistics
@st.cache_data
def get_publisher_stats(df_clean):
    """Calculate publisher average sales"""
    publisher_stats = df_clean.groupby('Publisher').agg({
        'Global_Sales': 'mean'
    }).reset_index()
    publisher_stats.columns = ['Publisher', 'Avg_Sales']
    return publisher_stats

# Get genre-platform combo statistics
@st.cache_data
def get_combo_stats(df_clean):
    """Calculate genre-platform combination average sales"""
    combo_stats = df_clean.groupby(['Genre', 'Platform']).agg({
        'Global_Sales': 'mean'
    }).reset_index()
    combo_stats.columns = ['Genre', 'Platform', 'Avg_Sales']
    return combo_stats

# Load data
with st.spinner("Loading and preprocessing data..."):
    data = load_and_preprocess_data(FEATURE_VERSION)

df = data['df']
df_clean = data['df_clean']
splits = data['splits']

# Page 1: Dataset Overview
if page == "📊 Dataset Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Hit Games", df_clean['Hit'].sum())
    with col4:
        st.metric("Flop Games", (df_clean['Hit'] == 0).sum())
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.bar_chart(missing[missing > 0])
    
    st.subheader("Correlation Heatmap")
    corr_matrix = dc.get_correlation_matrix(df_clean)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
    st.pyplot(fig)

# Page 2: Data Preprocessing
elif page == "🔧 Data Preprocessing":
    st.header("Data Preprocessing")
    
    st.subheader("Data Cleaning Steps")
    st.write("""
    1. **Missing Values**: Removed rows with missing Year, filled Publisher missing values
    2. **Duplicates**: Removed duplicate entries
    3. **Text Standardization**: Stripped whitespace from Genre, Platform, Publisher
    4. **Feature Engineering**: Created Hit/Flop binary target and Game_Age feature
    5. **Encoding**: Applied Label Encoding to Genre and Platform
    6. **Scaling**: Standardized features using StandardScaler
    """)
    
    st.subheader("Cleaned Dataset Shape")
    st.write(f"Original: {df.shape[0]} rows × {df.shape[1]} columns")
    st.write(f"Cleaned: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
    st.write(f"Rows removed: {df.shape[0] - df_clean.shape[0]}")
    
    st.subheader("Feature Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Genre Distribution**")
        genre_counts = df_clean['Genre'].value_counts().head(10)
        st.bar_chart(genre_counts)
    
    with col2:
        st.write("**Platform Distribution**")
        platform_counts = df_clean['Platform'].value_counts().head(10)
        st.bar_chart(platform_counts)
    
    st.subheader("Train-Test Split")
    st.write(f"**Regression**: Train {splits['X_reg_train'].shape[0]} | Test {splits['X_reg_test'].shape[0]}")
    st.write(f"**Classification**: Train {splits['X_clf_train'].shape[0]} | Test {splits['X_clf_test'].shape[0]}")

# Page 3: Regression Models
elif page == "📈 Regression Models":
    st.header("Regression Models")
    
    model_type = st.selectbox("Select Regression Model", 
                              ["Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression"])
    
    if st.button("Run Model"):
        with st.spinner("Training model..."):
            if model_type == "Simple Linear Regression":
                results = regression.simple_linear_regression(
                    splits['X_reg_train'], splits['y_reg_train'],
                    splits['X_reg_test'], splits['y_reg_test']
                )
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{results['mae']:.4f}")
                with col2:
                    st.metric("MSE", f"{results['mse']:.4f}")
                with col3:
                    st.metric("RMSE", f"{results['rmse']:.4f}")
                with col4:
                    st.metric("R² Score", f"{results['r2']:.4f}")
                
                st.write(f"Coefficient: {results['coefficient']:.4f}")
                st.write(f"Intercept: {results['intercept']:.4f}")
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                X_test = splits['X_reg_test'][['NA_Sales']].values[:500]
                y_test = splits['y_reg_test'].iloc[:500]
                y_pred = results['predictions'][:500]
                
                ax.scatter(X_test, y_test, alpha=0.5, label='Actual')
                ax.plot(X_test, y_pred, 'r-', label='Predicted', linewidth=2)
                ax.set_xlabel('NA_Sales')
                ax.set_ylabel('Global_Sales')
                ax.set_title('Simple Linear Regression')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            elif model_type == "Multiple Linear Regression":
                results = regression.multiple_linear_regression(
                    splits['X_reg_train_scaled'], splits['y_reg_train'],
                    splits['X_reg_test_scaled'], splits['y_reg_test']
                )
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{results['mae']:.4f}")
                with col2:
                    st.metric("MSE", f"{results['mse']:.4f}")
                with col3:
                    st.metric("RMSE", f"{results['rmse']:.4f}")
                with col4:
                    st.metric("R² Score", f"{results['r2']:.4f}")
                
                # Feature coefficients
                st.subheader("Feature Coefficients")
                coeff_df = pd.DataFrame({
                    'Feature': data['reg_features'],
                    'Coefficient': results['coefficients']
                }).sort_values('Coefficient', key=abs, ascending=False)
                st.dataframe(coeff_df, use_container_width=True)
                
                # Visualization - Actual vs Predicted
                st.subheader("Actual vs Predicted")
                fig, ax = plt.subplots(figsize=(10, 6))
                y_test = splits['y_reg_test']
                y_pred = results['predictions']
                
                # Sample for visualization if too many points
                sample_size = min(500, len(y_test))
                indices = np.random.choice(len(y_test), sample_size, replace=False)
                
                ax.scatter(y_test.iloc[indices], y_pred[indices], alpha=0.5, label='Predictions')
                # Perfect prediction line
                min_val = min(y_test.iloc[indices].min(), y_pred[indices].min())
                max_val = max(y_test.iloc[indices].max(), y_pred[indices].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
                ax.set_xlabel('Actual Global Sales')
                ax.set_ylabel('Predicted Global Sales')
                ax.set_title('Multiple Linear Regression: Actual vs Predicted')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Feature coefficients bar chart
                st.subheader("Feature Coefficients Visualization")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.barh(coeff_df['Feature'], coeff_df['Coefficient'])
                ax.set_xlabel('Coefficient Value')
                ax.set_title('Feature Coefficients in Multiple Linear Regression')
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig)
            
            elif model_type == "Polynomial Regression":
                results = regression.polynomial_regression(
                    splits['X_reg_train'], splits['y_reg_train'],
                    splits['X_reg_test'], splits['y_reg_test'],
                    degree=3
                )
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{results['mae']:.4f}")
                with col2:
                    st.metric("MSE", f"{results['mse']:.4f}")
                with col3:
                    st.metric("RMSE", f"{results['rmse']:.4f}")
                with col4:
                    st.metric("R² Score", f"{results['r2']:.4f}")
                
                # Visualization - Polynomial Regression Curve
                st.subheader("Polynomial Regression Curve")
                X_poly_test = splits['X_reg_test'][['Year']].values
                y_test = splits['y_reg_test']
                y_pred = results['predictions']
                
                # Sort by Year for smooth curve
                sorted_indices = np.argsort(X_poly_test.flatten())
                X_sorted = X_poly_test[sorted_indices]
                y_test_sorted = y_test.iloc[sorted_indices]
                y_pred_sorted = y_pred[sorted_indices]
                
                # Sample for visualization
                sample_size = min(500, len(X_sorted))
                sample_indices = np.linspace(0, len(X_sorted)-1, sample_size, dtype=int)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.scatter(X_sorted[sample_indices], y_test_sorted.iloc[sample_indices], 
                          alpha=0.5, label='Actual', s=20)
                ax.plot(X_sorted[sample_indices], y_pred_sorted[sample_indices], 
                       'r-', label='Polynomial Prediction', linewidth=2)
                ax.set_xlabel('Year')
                ax.set_ylabel('Global Sales')
                ax.set_title('Polynomial Regression: Global Sales vs Year (Degree=3)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Actual vs Predicted scatter
                st.subheader("Actual vs Predicted")
                fig, ax = plt.subplots(figsize=(10, 6))
                sample_size = min(500, len(y_test))
                indices = np.random.choice(len(y_test), sample_size, replace=False)
                
                ax.scatter(y_test.iloc[indices], y_pred[indices], alpha=0.5, label='Predictions')
                min_val = min(y_test.iloc[indices].min(), y_pred[indices].min())
                max_val = max(y_test.iloc[indices].max(), y_pred[indices].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
                ax.set_xlabel('Actual Global Sales')
                ax.set_ylabel('Predicted Global Sales')
                ax.set_title('Polynomial Regression: Actual vs Predicted')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

# Page 4: Classification Models
elif page == "🎯 Classification Models":
    st.header("Classification Models")
    
    model_type = st.selectbox("Select Classification Model",
                              ["K-Nearest Neighbors", "Naïve Bayes", "Decision Tree", "SVM"])
    
    if st.button("Run Model"):
        with st.spinner("Training model..."):
            if model_type == "K-Nearest Neighbors":
                results = classification.knn_classifier(
                    splits['X_clf_train_scaled'], splits['y_clf_train'],
                    splits['X_clf_test_scaled'], splits['y_clf_test']
                )
            elif model_type == "Naïve Bayes":
                results = classification.naive_bayes_classifier(
                    splits['X_clf_train_scaled'], splits['y_clf_train'],
                    splits['X_clf_test_scaled'], splits['y_clf_test']
                )
            elif model_type == "Decision Tree":
                results = classification.decision_tree_classifier(
                    splits['X_clf_train_scaled'], splits['y_clf_train'],
                    splits['X_clf_test_scaled'], splits['y_clf_test']
                )
            elif model_type == "SVM":
                results = classification.svm_classifier(
                    splits['X_clf_train_scaled'], splits['y_clf_train'],
                    splits['X_clf_test_scaled'], splits['y_clf_test']
                )
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{results['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{results['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{results['recall']:.4f}")
            with col4:
                st.metric("F1 Score", f"{results['f1']:.4f}")
            
            col5, col6 = st.columns(2)
            with col5:
                st.metric("ROC-AUC", f"{results['roc_auc']:.4f}")
            with col6:
                st.metric("Log Loss", f"{results['log_loss']:.4f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
            # ROC Curve
            st.subheader("ROC Curve")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(results['fpr'], results['tpr'], label=f'{model_type} (AUC = {results["roc_auc"]:.3f})', linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Feature Importance (for Decision Tree)
            if model_type == "Decision Tree":
                st.subheader("Top Feature Importance")
                feat_imp_df = pd.DataFrame({
                    'Feature': data['clf_features'],
                    'Importance': results['feature_importance']
                }).sort_values('Importance', ascending=False).head(5)
                st.dataframe(feat_imp_df, use_container_width=True)

# Page 5: Clustering
elif page == "🔍 Clustering":
    st.header("Clustering Analysis")
    
    clustering_type = st.selectbox("Select Clustering Method", ["K-Means", "Hierarchical Clustering"])
    
    if clustering_type == "K-Means":
        st.subheader("K-Means Clustering")
        
        k_value = st.slider("Number of Clusters (K)", 2, 10, 5)
        
        if st.button("Run K-Means"):
            with st.spinner("Clustering data..."):
                X_cluster, cluster_features = clustering.prepare_clustering_data(df_clean)
                scaler_cluster = StandardScaler()
                X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
                
                results = clustering.kmeans_clustering(X_cluster_scaled, n_clusters=k_value)
                
                st.write(f"**Inertia**: {results['inertia']:.2f}")
                st.write(f"**Cluster Distribution**:")
                cluster_counts = pd.Series(results['cluster_labels']).value_counts().sort_index()
                st.bar_chart(cluster_counts)
                
                # Elbow Method
                st.subheader("Elbow Method")
                elbow_results = clustering.elbow_method(X_cluster_scaled)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(elbow_results['k_range'], elbow_results['inertias'], 'bo-', linewidth=2, markersize=8)
                ax.set_xlabel('Number of Clusters (K)')
                ax.set_ylabel('Inertia')
                ax.set_title('Elbow Method')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # PCA Visualization
                st.subheader("Cluster Visualization (PCA)")
                pca_vis = PCA(n_components=2)
                X_cluster_pca = pca_vis.fit_transform(X_cluster_scaled)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                scatter = ax.scatter(X_cluster_pca[:, 0], X_cluster_pca[:, 1], 
                                    c=results['cluster_labels'], cmap='viridis', alpha=0.6, s=50)
                ax.scatter(pca_vis.transform(results['centers'])[:, 0], 
                          pca_vis.transform(results['centers'])[:, 1], 
                          c='red', marker='X', s=200, label='Centroids', linewidths=2)
                ax.set_xlabel('First Principal Component')
                ax.set_ylabel('Second Principal Component')
                ax.set_title(f'K-Means Clustering (K={k_value})')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Cluster')
                st.pyplot(fig)
    
    elif clustering_type == "Hierarchical Clustering":
        st.subheader("Hierarchical Clustering")
        
        linkage_method = st.selectbox("Linkage Method", ["single", "complete", "average", "ward"])
        n_clusters = st.slider("Number of Clusters", 2, 10, 5)
        
        if st.button("Run Hierarchical Clustering"):
            with st.spinner("Clustering data..."):
                X_cluster, cluster_features = clustering.prepare_clustering_data(df_clean)
                scaler_cluster = StandardScaler()
                X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
                
                results = clustering.hierarchical_clustering(X_cluster_scaled, n_clusters=n_clusters, 
                                                linkage_method=linkage_method)
                
                st.write(f"**Cluster Distribution**:")
                cluster_counts = pd.Series(results['cluster_labels']).value_counts().sort_index()
                st.bar_chart(cluster_counts)
                
                # Dendrogram
                st.subheader("Dendrogram")
                fig, ax = plt.subplots(figsize=(12, 8))
                dendrogram(results['linkage_matrix'], ax=ax, truncate_mode='level', p=5)
                ax.set_title(f'Hierarchical Clustering - {linkage_method.capitalize()} Linkage')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Distance')
                st.pyplot(fig)

# Page 6: Neural Networks
elif page == "🧠 Neural Networks":
    st.header("Neural Networks")
    
    model_type = st.selectbox("Select Model", ["MLP Classifier", "MLP Regressor"])
    
    if st.button("Run Model"):
        with st.spinner("Training neural network..."):
            if model_type == "MLP Classifier":
                results = neural_networks.mlp_classifier(
                    splits['X_clf_train_scaled'], splits['y_clf_train'],
                    splits['X_clf_test_scaled'], splits['y_clf_test']
                )
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{results['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{results['recall']:.4f}")
                with col4:
                    st.metric("F1 Score", f"{results['f1']:.4f}")
                
                st.metric("ROC-AUC", f"{results['roc_auc']:.4f}")
                st.write(f"Iterations: {results['n_iter']}, Final Loss: {results['loss']:.4f}")
                
                # Loss curve if available
                if hasattr(results['model'], 'loss_curve_'):
                    st.subheader("Training Loss Curve")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(results['model'].loss_curve_, linewidth=2)
                    ax.set_xlabel('Iterations')
                    ax.set_ylabel('Loss')
                    ax.set_title('MLP Classifier - Training Loss')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
            elif model_type == "MLP Regressor":
                results = neural_networks.mlp_regressor(
                    splits['X_reg_train_scaled'], splits['y_reg_train'],
                    splits['X_reg_test_scaled'], splits['y_reg_test']
                )
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{results['mae']:.4f}")
                with col2:
                    st.metric("MSE", f"{results['mse']:.4f}")
                with col3:
                    st.metric("RMSE", f"{results['rmse']:.4f}")
                with col4:
                    st.metric("R² Score", f"{results['r2']:.4f}")
                
                st.write(f"Iterations: {results['n_iter']}, Final Loss: {results['loss']:.4f}")
                
                # Actual vs Predicted scatter
                st.subheader("Actual vs Predicted")
                fig, ax = plt.subplots(figsize=(10, 6))
                y_test = splits['y_reg_test']
                y_pred = results['predictions']
                
                # Sample for visualization
                sample_size = min(500, len(y_test))
                indices = np.random.choice(len(y_test), sample_size, replace=False)
                
                ax.scatter(y_test.iloc[indices], y_pred[indices], alpha=0.5, label='Predictions')
                min_val = min(y_test.iloc[indices].min(), y_pred[indices].min())
                max_val = max(y_test.iloc[indices].max(), y_pred[indices].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
                ax.set_xlabel('Actual Global Sales')
                ax.set_ylabel('Predicted Global Sales')
                ax.set_title('MLP Regressor: Actual vs Predicted')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Loss curve if available
                if hasattr(results['model'], 'loss_curve_'):
                    st.subheader("Training Loss Curve")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(results['model'].loss_curve_, linewidth=2)
                    ax.set_xlabel('Iterations')
                    ax.set_ylabel('Loss')
                    ax.set_title('MLP Regressor - Training Loss Curve')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

# Page 7: Ensemble Methods
elif page == "⚡ Ensemble Methods":
    st.header("Ensemble Methods")
    
    model_type = st.selectbox("Select Ensemble Method", 
                              ["Bagging", "AdaBoost", "Random Forest"])
    
    if st.button("Run Model"):
        with st.spinner("Training ensemble model..."):
            if model_type == "Bagging":
                results = ensemble.bagging_classifier(
                    splits['X_clf_train_scaled'], splits['y_clf_train'],
                    splits['X_clf_test_scaled'], splits['y_clf_test']
                )
            elif model_type == "AdaBoost":
                results = ensemble.adaboost_classifier(
                    splits['X_clf_train_scaled'], splits['y_clf_train'],
                    splits['X_clf_test_scaled'], splits['y_clf_test']
                )
            elif model_type == "Random Forest":
                results = ensemble.random_forest_classifier(
                    splits['X_clf_train_scaled'], splits['y_clf_train'],
                    splits['X_clf_test_scaled'], splits['y_clf_test']
                )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{results['accuracy']:.4f}")
            with col2:
                st.metric("ROC-AUC", f"{results['roc_auc']:.4f}")
            
            # Feature Importance for Random Forest
            if model_type == "Random Forest":
                st.subheader("Top Feature Importance")
                feat_imp_df = pd.DataFrame({
                    'Feature': data['clf_features'],
                    'Importance': results['feature_importance']
                }).sort_values('Importance', ascending=False).head(5)
                st.dataframe(feat_imp_df, use_container_width=True)

# Page 8: Model Comparison
elif page == "📋 Model Comparison":
    st.header("Model Comparison")
    
    comparison_type = st.selectbox("Select Comparison Type",
                                   ["Regression Models", "Classification Models", "Ensemble Methods"])
    
    if st.button("Generate Comparison"):
        with st.spinner("Running all models and comparing..."):
            if comparison_type == "Regression Models":
                # Run all regression models
                simple_results = regression.simple_linear_regression(
                    splits['X_reg_train'], splits['y_reg_train'],
                    splits['X_reg_test'], splits['y_reg_test']
                )
                
                multiple_results = regression.multiple_linear_regression(
                    splits['X_reg_train_scaled'], splits['y_reg_train'],
                    splits['X_reg_test_scaled'], splits['y_reg_test']
                )
                
                poly_results = regression.polynomial_regression(
                    splits['X_reg_train'], splits['y_reg_train'],
                    splits['X_reg_test'], splits['y_reg_test']
                )
                
                comparison_df = pd.DataFrame({
                    'Model': ['Simple LR', 'Multiple LR', 'Polynomial LR'],
                    'MAE': [simple_results['mae'], multiple_results['mae'], poly_results['mae']],
                    'MSE': [simple_results['mse'], multiple_results['mse'], poly_results['mse']],
                    'RMSE': [simple_results['rmse'], multiple_results['rmse'], poly_results['rmse']],
                    'R² Score': [simple_results['r2'], multiple_results['r2'], poly_results['r2']]
                })
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualization
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                metrics = ['MAE', 'MSE', 'RMSE', 'R² Score']
                for idx, metric in enumerate(metrics):
                    ax = axes[idx // 2, idx % 2]
                    ax.bar(comparison_df['Model'], comparison_df[metric])
                    ax.set_title(f'{metric} Comparison')
                    ax.set_ylabel(metric)
                    ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            elif comparison_type == "Classification Models":
                # Run all classification models
                knn_results = classification.knn_classifier(
                    splits['X_clf_train_scaled'], splits['y_clf_train'],
                    splits['X_clf_test_scaled'], splits['y_clf_test']
                )
                
                nb_results = classification.naive_bayes_classifier(
                    splits['X_clf_train_scaled'], splits['y_clf_train'],
                    splits['X_clf_test_scaled'], splits['y_clf_test']
                )
                
                dt_results = classification.decision_tree_classifier(
                    splits['X_clf_train_scaled'], splits['y_clf_train'],
                    splits['X_clf_test_scaled'], splits['y_clf_test']
                )
                
                svm_results = classification.svm_classifier(
                    splits['X_clf_train_scaled'], splits['y_clf_train'],
                    splits['X_clf_test_scaled'], splits['y_clf_test']
                )
                
                comparison_df = pd.DataFrame({
                    'Model': ['KNN', 'Naïve Bayes', 'Decision Tree', 'SVM'],
                    'Accuracy': [knn_results['accuracy'], nb_results['accuracy'], 
                               dt_results['accuracy'], svm_results['accuracy']],
                    'Precision': [knn_results['precision'], nb_results['precision'],
                                 dt_results['precision'], svm_results['precision']],
                    'Recall': [knn_results['recall'], nb_results['recall'],
                              dt_results['recall'], svm_results['recall']],
                    'F1 Score': [knn_results['f1'], nb_results['f1'],
                               dt_results['f1'], svm_results['f1']],
                    'ROC-AUC': [knn_results['roc_auc'], nb_results['roc_auc'],
                               dt_results['roc_auc'], svm_results['roc_auc']]
                })
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # ROC Curves Comparison
                st.subheader("ROC Curves Comparison")
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(knn_results['fpr'], knn_results['tpr'], 
                       label=f'KNN (AUC = {knn_results["roc_auc"]:.3f})', linewidth=2)
                ax.plot(nb_results['fpr'], nb_results['tpr'], 
                       label=f'Naïve Bayes (AUC = {nb_results["roc_auc"]:.3f})', linewidth=2)
                ax.plot(dt_results['fpr'], dt_results['tpr'], 
                       label=f'Decision Tree (AUC = {dt_results["roc_auc"]:.3f})', linewidth=2)
                ax.plot(svm_results['fpr'], svm_results['tpr'], 
                       label=f'SVM (AUC = {svm_results["roc_auc"]:.3f})', linewidth=2)
                ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curves - Classification Models')
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            elif comparison_type == "Ensemble Methods":
                # Run all ensemble methods
                bagging_results = ensemble.bagging_classifier(
                    splits['X_clf_train_scaled'], splits['y_clf_train'],
                    splits['X_clf_test_scaled'], splits['y_clf_test']
                )
                
                adaboost_results = ensemble.adaboost_classifier(
                    splits['X_clf_train_scaled'], splits['y_clf_train'],
                    splits['X_clf_test_scaled'], splits['y_clf_test']
                )
                
                rf_results = ensemble.random_forest_classifier(
                    splits['X_clf_train_scaled'], splits['y_clf_train'],
                    splits['X_clf_test_scaled'], splits['y_clf_test']
                )
                
                comparison_df = pd.DataFrame({
                    'Model': ['Bagging', 'AdaBoost', 'Random Forest'],
                    'Accuracy': [bagging_results['accuracy'], adaboost_results['accuracy'], 
                               rf_results['accuracy']],
                    'ROC-AUC': [bagging_results['roc_auc'], adaboost_results['roc_auc'],
                               rf_results['roc_auc']]
                })
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualization
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                axes[0].bar(comparison_df['Model'], comparison_df['Accuracy'])
                axes[0].set_title('Accuracy Comparison')
                axes[0].set_ylabel('Accuracy')
                axes[0].tick_params(axis='x', rotation=45)
                
                axes[1].bar(comparison_df['Model'], comparison_df['ROC-AUC'])
                axes[1].set_title('ROC-AUC Comparison')
                axes[1].set_ylabel('ROC-AUC')
                axes[1].tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

# Page 9: Game Prediction
elif page == "🎮 Game Prediction":
    st.header("🎮 Game Success Prediction")
    
    # Check if models are using the correct feature count
    expected_features = len(data['clf_features'])
    if expected_features != 13:
        st.warning(f"""
        ⚠️ **Model Feature Mismatch Detected**
        
        The models appear to be using {expected_features} features instead of 13. 
        This can happen if the app was started before the feature update.
        
        **Solution:** 
        1. Click "🔄 Clear Cache & Reload" in the sidebar, OR
        2. Restart the Streamlit app completely (stop with Ctrl+C and run `streamlit run app.py` again)
        
        This will retrain the models with all 13 features including Publisher and derived features.
        """)
    
    st.markdown("""
    Predict whether a new game will be a **Hit** (≥1M global sales) or **Flop** (<1M global sales) 
    based on its characteristics. The model uses **13 features** including:
    - **Basic Info**: Genre, Platform, Publisher, Release Year
    - **Regional Sales**: NA, EU, JP, and Other regions (estimated or custom)
    - **Derived Features**: Publisher track record, Genre-Platform popularity, Total regional sales, Year trends
    
    You can either provide regional sales estimates or let the system use historical averages for similar games.
    """)
    
    # Train models if not already cached
    with st.spinner("Loading prediction models..."):
        prediction_models = train_prediction_models(splits, FEATURE_VERSION)
        avg_sales = get_average_regional_sales(df_clean)
        publisher_stats = get_publisher_stats(df_clean)
        combo_stats = get_combo_stats(df_clean)
    
    # Get unique genres, platforms, and publishers
    genres = sorted(df_clean['Genre'].unique())
    platforms = sorted(df_clean['Platform'].unique())
    publishers = sorted(df_clean['Publisher'].unique())
    
    # Input form
    st.subheader("Game Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        game_name = st.text_input("Game Name (optional)", placeholder="Enter game name")
        release_year = st.number_input("Release Year", min_value=1980, max_value=2030, value=2024, step=1)
        genre = st.selectbox("Genre", genres)
        platform = st.selectbox("Platform", platforms)
    
    with col2:
        publisher = st.selectbox("Publisher", publishers, help="Select the game publisher. Publisher track record affects prediction accuracy.")
        use_custom_sales = st.checkbox("Provide custom regional sales estimates", value=False)
        
        if use_custom_sales:
            st.write("**Regional Sales Estimates (in millions)**")
            na_sales = st.number_input("North America Sales", min_value=0.0, value=0.0, step=0.1, format="%.2f")
            eu_sales = st.number_input("Europe Sales", min_value=0.0, value=0.0, step=0.1, format="%.2f")
            jp_sales = st.number_input("Japan Sales", min_value=0.0, value=0.0, step=0.1, format="%.2f")
            other_sales = st.number_input("Other Regions Sales", min_value=0.0, value=0.0, step=0.1, format="%.2f")
        else:
            st.info("💡 Regional sales will be estimated using historical averages for similar games (same Genre + Platform)")
            na_sales = None
            eu_sales = None
            jp_sales = None
            other_sales = None
    
    with col3:
        st.write("**Additional Information**")
        # Show publisher average sales if available
        pub_avg = publisher_stats[publisher_stats['Publisher'] == publisher]
        if len(pub_avg) > 0:
            st.metric("Publisher Avg Sales", f"{pub_avg['Avg_Sales'].iloc[0]:.2f}M", 
                     help="Average global sales for games from this publisher")
        
        # Show genre-platform combo average
        combo_avg = combo_stats[(combo_stats['Genre'] == genre) & (combo_stats['Platform'] == platform)]
        if len(combo_avg) > 0:
            st.metric("Genre-Platform Avg", f"{combo_avg['Avg_Sales'].iloc[0]:.2f}M",
                     help="Average global sales for this genre-platform combination")
    
    # Predict button
    if st.button("🔮 Predict Game Success", type="primary", use_container_width=True):
        with st.spinner("Analyzing game data and making prediction..."):
            # Calculate Game_Age
            current_year = 2024  # You can update this
            game_age = current_year - release_year
            
            # Get or estimate regional sales
            if use_custom_sales:
                estimated_na = na_sales
                estimated_eu = eu_sales
                estimated_jp = jp_sales
                estimated_other = other_sales
            else:
                # Use historical averages for this genre/platform combination
                matching_sales = avg_sales[
                    (avg_sales['Genre'] == genre) & 
                    (avg_sales['Platform'] == platform)
                ]
                
                if len(matching_sales) > 0:
                    estimated_na = matching_sales['NA_Sales'].iloc[0]
                    estimated_eu = matching_sales['EU_Sales'].iloc[0]
                    estimated_jp = matching_sales['JP_Sales'].iloc[0]
                    estimated_other = matching_sales['Other_Sales'].iloc[0]
                else:
                    # Fallback to overall averages if no match
                    estimated_na = df_clean['NA_Sales'].mean()
                    estimated_eu = df_clean['EU_Sales'].mean()
                    estimated_jp = df_clean['JP_Sales'].mean()
                    estimated_other = df_clean['Other_Sales'].mean()
            
            # Encode genre, platform, and publisher
            try:
                genre_encoded = data['le_genre'].transform([genre])[0]
                platform_encoded = data['le_platform'].transform([platform])[0]
                publisher_encoded = data['le_publisher'].transform([publisher])[0]
            except:
                # If new genre/platform/publisher not in training data, use 0
                genre_encoded = 0
                platform_encoded = 0
                publisher_encoded = 0
            
            # Calculate additional features
            total_regional_sales = estimated_na + estimated_eu + estimated_jp + estimated_other
            
            # Get publisher average sales
            pub_avg_sales = publisher_stats[publisher_stats['Publisher'] == publisher]
            if len(pub_avg_sales) > 0:
                publisher_avg_sales = pub_avg_sales['Avg_Sales'].iloc[0]
            else:
                publisher_avg_sales = df_clean['Global_Sales'].mean()
            
            # Get genre-platform combo average sales
            combo_avg_sales = combo_stats[(combo_stats['Genre'] == genre) & (combo_stats['Platform'] == platform)]
            if len(combo_avg_sales) > 0:
                combo_avg = combo_avg_sales['Avg_Sales'].iloc[0]
            else:
                combo_avg = df_clean['Global_Sales'].mean()
            
            # Calculate normalized year (0 to 1 scale)
            year_min = df_clean['Year'].min()
            year_max = df_clean['Year'].max()
            year_normalized = (release_year - year_min) / (year_max - year_min) if year_max > year_min else 0.5
            
            # Prepare feature vector (13 features total)
            features = np.array([[
                estimated_na,                    # 0: NA_Sales
                estimated_eu,                    # 1: EU_Sales
                estimated_jp,                    # 2: JP_Sales
                estimated_other,                 # 3: Other_Sales
                release_year,                    # 4: Year
                genre_encoded,                   # 5: Genre_Encoded
                platform_encoded,                # 6: Platform_Encoded
                publisher_encoded,               # 7: Publisher_Encoded
                game_age,                        # 8: Game_Age
                total_regional_sales,           # 9: Total_Regional_Sales
                publisher_avg_sales,            # 10: Publisher_Avg_Sales
                combo_avg,                       # 11: Combo_Avg_Sales
                year_normalized                  # 12: Year_Normalized
            ]])
            
            # Verify feature count matches model expectations
            expected_feature_count = len(data['clf_features'])
            actual_feature_count = features.shape[1]
            
            if actual_feature_count != expected_feature_count:
                st.error(f"""
                **Feature Count Mismatch Error!**
                
                - Expected: {expected_feature_count} features
                - Provided: {actual_feature_count} features
                
                This usually happens when the model cache needs to be refreshed. 
                Please restart the Streamlit app to retrain models with the new feature set.
                
                **To fix:** Stop the app (Ctrl+C) and restart it with `streamlit run app.py`
                """)
                st.stop()
            
            # Scale features for classification
            features_scaled_clf = prediction_models['clf_scaler'].transform(features)
            
            # Predict Hit/Flop
            hit_prediction = prediction_models['clf_model'].predict(features_scaled_clf)[0]
            hit_probability = prediction_models['clf_model'].predict_proba(features_scaled_clf)[0]
            
            # Predict Global Sales
            features_scaled_reg = prediction_models['reg_scaler'].transform(features)
            predicted_global_sales = prediction_models['reg_model'].predict(features_scaled_reg)[0]
            predicted_global_sales = max(0, predicted_global_sales)  # Ensure non-negative
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Main prediction card
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                if hit_prediction == 1:
                    st.markdown("""
                    <div style='background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;'>
                        <h2 style='color: #155724; margin: 0;'>🎯 HIT GAME!</h2>
                        <p style='color: #155724; margin-top: 10px; font-size: 1.1em;'>
                            This game is predicted to be a <strong>HIT</strong> with ≥1M global sales!
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545;'>
                        <h2 style='color: #721c24; margin: 0;'>⚠️ FLOP GAME</h2>
                        <p style='color: #721c24; margin-top: 10px; font-size: 1.1em;'>
                            This game is predicted to be a <strong>FLOP</strong> with <1M global sales.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                confidence = hit_probability[1] if hit_prediction == 1 else hit_probability[0]
                st.metric("Confidence", f"{confidence*100:.1f}%")
            
            with col3:
                st.metric("Predicted Global Sales", f"{predicted_global_sales:.2f}M")
            
            # Detailed metrics
            st.markdown("---")
            st.subheader("📈 Detailed Predictions")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Predicted NA Sales", f"{estimated_na:.2f}M")
            with col2:
                st.metric("Predicted EU Sales", f"{estimated_eu:.2f}M")
            with col3:
                st.metric("Predicted JP Sales", f"{estimated_jp:.2f}M")
            with col4:
                st.metric("Predicted Other Sales", f"{estimated_other:.2f}M")
            
            # Probability breakdown
            st.markdown("---")
            st.subheader("🎲 Probability Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Probability bar chart
                prob_df = pd.DataFrame({
                    'Outcome': ['Flop (<1M)', 'Hit (≥1M)'],
                    'Probability': [hit_probability[0]*100, hit_probability[1]*100]
                })
                
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ['#dc3545' if hit_prediction == 0 else '#f8d7da', 
                          '#28a745' if hit_prediction == 1 else '#d4edda']
                bars = ax.bar(prob_df['Outcome'], prob_df['Probability'], color=colors)
                ax.set_ylabel('Probability (%)')
                ax.set_title('Success Probability')
                ax.set_ylim(0, 100)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                ax.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig)
            
            with col2:
                # Feature importance (if available)
                st.write("**Key Factors**")
                st.write(f"• **Genre**: {genre}")
                st.write(f"• **Platform**: {platform}")
                st.write(f"• **Publisher**: {publisher}")
                st.write(f"• **Release Year**: {release_year}")
                st.write(f"• **Game Age**: {game_age} years")
                st.write(f"• **Total Regional Sales**: {total_regional_sales:.2f}M")
                st.write(f"• **Publisher Avg**: {publisher_avg_sales:.2f}M")
                st.write(f"• **Combo Avg**: {combo_avg:.2f}M")
                
                if not use_custom_sales:
                    st.info("💡 Regional sales were estimated from historical data")
            
            # Insights and recommendations
            st.markdown("---")
            st.subheader("💡 Insights & Recommendations")
            
            if hit_prediction == 1:
                if predicted_global_sales >= 5.0:
                    st.success("🌟 **Excellent Potential!** This game shows strong potential for exceptional sales performance.")
                elif predicted_global_sales >= 2.0:
                    st.success("✅ **Good Potential!** This game is likely to perform well in the market.")
                else:
                    st.info("📊 **Moderate Success** - The game is predicted to cross the 1M threshold, but may need marketing support.")
            else:
                if predicted_global_sales < 0.5:
                    st.warning("⚠️ **High Risk** - This game faces significant challenges. Consider:")
                    st.write("  - Market research to understand target audience better")
                    st.write("  - Enhanced marketing strategy")
                    st.write("  - Platform/genre combination optimization")
                else:
                    st.info("📉 **Moderate Risk** - The game may need strategic improvements:")
                    st.write("  - Strong marketing campaign")
                    st.write("  - Consider different platform or release timing")
                    st.write("  - Focus on key markets (NA/EU)")
            
            # Comparison with similar games
            st.markdown("---")
            st.subheader("📊 Comparison with Similar Games")
            
            # Similar games by genre and platform
            similar_games = df_clean[
                (df_clean['Genre'] == genre) & 
                (df_clean['Platform'] == platform)
            ]
            
            # Games by same publisher
            publisher_games = df_clean[df_clean['Publisher'] == publisher]
            
            if len(similar_games) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_sales_similar = similar_games['Global_Sales'].mean()
                    st.metric("Avg Sales (Genre+Platform)", f"{avg_sales_similar:.2f}M")
                
                with col2:
                    hit_rate = (similar_games['Hit'] == 1).mean() * 100
                    st.metric("Hit Rate (Genre+Platform)", f"{hit_rate:.1f}%")
                
                with col3:
                    if len(publisher_games) > 0:
                        pub_avg_sales = publisher_games['Global_Sales'].mean()
                        pub_hit_rate = (publisher_games['Hit'] == 1).mean() * 100
                        st.metric("Publisher Avg Sales", f"{pub_avg_sales:.2f}M")
                    else:
                        st.metric("Publisher Avg Sales", "N/A")
                
                with col4:
                    total_similar = len(similar_games)
                    st.metric("Similar Games (Genre+Platform)", total_similar)
                
                # Show if prediction is above/below average
                if predicted_global_sales > avg_sales_similar:
                    st.success(f"✅ Your game is predicted to perform **{((predicted_global_sales/avg_sales_similar - 1)*100):.1f}% better** than average similar games (Genre+Platform)!")
                else:
                    st.warning(f"⚠️ Your game is predicted to perform **{((1 - predicted_global_sales/avg_sales_similar)*100):.1f}% worse** than average similar games (Genre+Platform).")
                
                # Publisher comparison
                if len(publisher_games) > 0:
                    if predicted_global_sales > pub_avg_sales:
                        st.info(f"📈 Compared to {publisher}'s average: **{((predicted_global_sales/pub_avg_sales - 1)*100):.1f}% better** (Publisher hit rate: {pub_hit_rate:.1f}%)")
                    else:
                        st.info(f"📉 Compared to {publisher}'s average: **{((1 - predicted_global_sales/pub_avg_sales)*100):.1f}% worse** (Publisher hit rate: {pub_hit_rate:.1f}%)")
            else:
                st.info("No similar games found in the dataset for comparison.")

#  footer
