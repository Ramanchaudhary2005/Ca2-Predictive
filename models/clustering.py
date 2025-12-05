"""
Clustering Models Module
Implements K-Means and Hierarchical Clustering
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage


def kmeans_clustering(X_cluster_scaled, n_clusters=5, random_state=42):
    """K-Means Clustering"""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = model.fit_predict(X_cluster_scaled)
    
    return {
        'model': model,
        'cluster_labels': cluster_labels,
        'inertia': model.inertia_,
        'centers': model.cluster_centers_
    }


def elbow_method(X_cluster_scaled, k_range=range(2, 11)):
    """Elbow Method to find optimal K"""
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_cluster_scaled)
        inertias.append(kmeans.inertia_)
    
    return {
        'k_range': list(k_range),
        'inertias': inertias
    }


def hierarchical_clustering(X_cluster_scaled, n_clusters=5, linkage_method='average', sample_size=500):
    """Hierarchical Clustering (Agglomerative)"""
    # Use sample for faster computation
    if len(X_cluster_scaled) > sample_size:
        indices = np.random.choice(len(X_cluster_scaled), sample_size, replace=False)
        X_sample = X_cluster_scaled[indices]
    else:
        X_sample = X_cluster_scaled
        indices = np.arange(len(X_cluster_scaled))
    
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    cluster_labels = model.fit_predict(X_sample)
    
    # Create linkage matrix for dendrogram
    linkage_matrix = linkage(X_sample, method=linkage_method)
    
    return {
        'model': model,
        'cluster_labels': cluster_labels,
        'linkage_matrix': linkage_matrix,
        'sample_indices': indices
    }


def prepare_clustering_data(df_clean):
    """Prepare data for clustering"""
    clustering_features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Year', 'Genre_Encoded', 'Platform_Encoded']
    X_cluster = df_clean[clustering_features].dropna()
    return X_cluster, clustering_features

