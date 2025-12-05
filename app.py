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
st.markdown('<h1 class="main-header">🎮 Predictive Analytics: Video Games Sales Analysis</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Dataset Overview", "🔧 Data Preprocessing", "📈 Regression Models", 
     "🎯 Classification Models", "🔍 Clustering", "🧠 Neural Networks", 
     "⚡ Ensemble Methods", "📋 Model Comparison"]
)

# Load data (cached)
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess data"""
    df = dc.load_data('video_games_sales.csv')
    df_clean = dc.clean_data(df)
    df_clean = dc.engineer_features(df_clean)
    df_clean, le_genre, le_platform = dc.encode_categorical(df_clean)
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
        'le_platform': le_platform
    }

# Load data
with st.spinner("Loading and preprocessing data..."):
    data = load_and_preprocess_data()

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

# Footer


