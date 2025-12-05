# Predictive Analytics Project: Video Games Sales Analysis

## 📋 Project Overview

This project demonstrates comprehensive predictive analytics techniques using video games sales data. The project covers all six units of the Predictive Analytics syllabus, implementing various machine learning algorithms including regression, classification, clustering, dimensionality reduction, and neural networks.

## 🎯 Project Objectives

- Implement and compare various regression models (Simple, Multiple, Polynomial, Logistic)
- Apply classification algorithms (KNN, Naïve Bayes, Decision Trees, SVM)
- Perform unsupervised learning using clustering techniques (K-Means, Hierarchical)
- Conduct market basket analysis using association rules
- Apply dimensionality reduction using PCA
- Implement neural networks (MLP)
- Evaluate model performance using cross-validation and ensemble methods

## 📊 Dataset

**Dataset:** `video_games_sales.csv`

**Description:** Contains video game sales data with the following features:
- **Rank**: Ranking of overall sales
- **Name**: Name of the game
- **Platform**: Platform on which the game was released
- **Year**: Year of release
- **Genre**: Genre of the game
- **Publisher**: Publisher of the game
- **NA_Sales**: Sales in North America (in millions)
- **EU_Sales**: Sales in Europe (in millions)
- **JP_Sales**: Sales in Japan (in millions)
- **Other_Sales**: Sales in other regions (in millions)
- **Global_Sales**: Total worldwide sales (in millions)

## 🛠️ Requirements

### Python Libraries

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
streamlit
jupyter
```

### Installation

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
Ca2 Predictive/
│
├── video_games_sales.csv              # Dataset file
├── Predictive_Analytics_Project.ipynb  # Main project notebook
├── app.py                              # Streamlit web application
├── data_cleaning.py                   # Data preprocessing module
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── models/                             # Machine learning models
    ├── __init__.py
    ├── regression.py                  # Regression models
    ├── classification.py              # Classification models
    ├── clustering.py                  # Clustering models
    ├── neural_networks.py             # Neural network models
    └── ensemble.py                    # Ensemble methods
```

## 🚀 How to Run

### Option 1: Streamlit Web Application (Recommended)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Access the web interface:**
   - The app will automatically open in your default web browser
   - If not, navigate to `http://localhost:8501`
   - Use the sidebar to navigate between different sections

### Option 2: Jupyter Notebook

1. **Ensure all dependencies are installed:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook Predictive_Analytics_Project.ipynb
   ```

3. **Run all cells sequentially:**
   - The notebook is designed to run from top to bottom
   - Each cell builds upon previous cells
   - Make sure the dataset file `video_games_sales.csv` is in the same directory

## 📚 Project Contents

### Unit I: Introduction and Data Preparation

- **Data Loading & Exploration**: Loading dataset, checking shape, info, and statistics
- **Data Cleaning**: Handling missing values, removing duplicates, standardizing text
- **Feature Engineering**: Creating binary classification target (Hit/Flop), game age feature
- **Encoding**: Label encoding for categorical variables (Genre, Platform)
- **Train-Test Split**: 80-20 split for both regression and classification tasks
- **Data Scaling**: StandardScaler for normalization
- **Correlation Analysis**: Heatmap visualization of feature correlations

### Unit II: Supervised Learning - REGRESSION

#### 2.1 Simple Linear Regression
- Predicts Global_Sales from NA_Sales
- **Metrics**: MAE, MSE, RMSE, R² Score
- Visualization: Scatter plot with regression line

#### 2.2 Multiple Linear Regression
- Uses all features (NA_Sales, EU_Sales, JP_Sales, Other_Sales, Year, Genre_Encoded, Platform_Encoded, Game_Age)
- **Metrics**: MAE, MSE, RMSE, R² Score
- Feature coefficient analysis

#### 2.3 Polynomial Regression
- Degree 3 polynomial regression using Year as feature
- **Metrics**: MAE, MSE, RMSE, R² Score
- Visualization: Polynomial curve fitting

#### 2.4 Logistic Regression
- Binary classification (Hit/Flop prediction)
- **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC

#### 2.5 Regression Models Comparison
- Comparative analysis of all regression models
- Visualization: Bar charts for all metrics

### Unit III: Supervised Learning - CLASSIFICATION

#### 3.1 K-Nearest Neighbors (KNN)
- Lazy learning algorithm
- **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC, Log Loss

#### 3.2 Naïve Bayes
- Probabilistic classifier
- **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC, Log Loss

#### 3.3 Decision Trees
- Divide and conquer approach
- **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC, Log Loss
- Feature importance analysis

#### 3.4 Support Vector Machine (SVM)
- RBF kernel SVM
- **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC, Log Loss

#### 3.5 Classification Models Comparison
- ROC curves for all models
- Comparative bar charts for key metrics

### Unit IV: Unsupervised Learning - CLUSTERING AND PATTERN DETECTION

#### 4.1 K-Means Clustering
- Elbow method for optimal K selection
- K-Means clustering with K=5
- **Visualization**: PCA projection of clusters with centroids

#### 4.2 Hierarchical Clustering
- Agglomerative clustering
- **Linkage Methods**: Single, Complete, Average, Ward
- **Visualization**: Dendrograms for all linkage methods
- Implementation with Average linkage

#### 4.3 Association Rules - Market Basket Analysis
- Market basket analysis using Genre and Platform
- **Metrics**: Support, Confidence, Lift
- **Visualization**: Top 10 association rules by Lift

### Unit V: Dimensionality Reduction and Neural Networks

#### 5.1 Principal Component Analysis (PCA)
- Explained variance analysis
- Cumulative variance plot
- **Visualization**: 2D PCA space colored by Hit/Flop

#### 5.2 Multi-Layer Perceptron (MLP)
- **MLP Classifier**: Binary classification
  - Architecture: (100, 50) hidden layers
  - **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **MLP Regressor**: Sales prediction
  - **Metrics**: MAE, MSE, RMSE, R² Score
- Training loss curve visualization

### Unit VI: Model Performance

#### 6.1 Bias-Variance Trade-off
- Analysis using Decision Trees with varying depths (1-20)
- **Visualization**: Training vs Test accuracy plot
- Identification of optimal depth

#### 6.2 Cross-Validation Methods
- **K-Fold Cross-Validation**: K=5
  - Applied to: Logistic Regression, Decision Tree, KNN, Naïve Bayes
- **Leave-One-Out Cross-Validation**
  - Applied to: Logistic Regression, KNN
- **Visualization**: Box plots for CV results

#### 6.3 Ensemble Methods
- **Bagging**: BaggingClassifier with Decision Trees
- **Boosting**: AdaBoostClassifier
- **Random Forest**: RandomForestClassifier
- **Metrics**: Accuracy, ROC-AUC
- **Visualization**: Comparative bar charts

#### 6.4 Final Model Performance Summary
- Comprehensive summary of all models
- Comparison tables for regression, classification, ensemble methods, and neural networks

## 📈 Key Results

The project demonstrates:

1. **Regression Performance**: Multiple Linear Regression typically performs best among regression models
2. **Classification Performance**: Ensemble methods (Random Forest, Bagging, AdaBoost) show superior performance
3. **Clustering**: K-Means identifies 5 distinct clusters in the data
4. **Dimensionality Reduction**: PCA shows that first few components capture most variance
5. **Neural Networks**: MLP achieves competitive performance for both classification and regression tasks

## 🌐 Streamlit Web Application Features

The Streamlit app provides an interactive web interface with the following pages:

1. **📊 Dataset Overview**: View dataset statistics, preview, missing values, and correlation heatmap
2. **🔧 Data Preprocessing**: See data cleaning steps, feature distributions, and train-test split information
3. **📈 Regression Models**: Run and compare Simple Linear, Multiple Linear, and Polynomial Regression
4. **🎯 Classification Models**: Test KNN, Naïve Bayes, Decision Tree, and SVM classifiers
5. **🔍 Clustering**: Perform K-Means and Hierarchical Clustering with visualizations
6. **🧠 Neural Networks**: Train MLP Classifier and Regressor
7. **⚡ Ensemble Methods**: Compare Bagging, AdaBoost, and Random Forest
8. **📋 Model Comparison**: Side-by-side comparison of all models with visualizations

### Streamlit App Features:
- ✅ Interactive model selection and execution
- ✅ Real-time metrics display
- ✅ Visualizations (charts, heatmaps, ROC curves)
- ✅ Model comparison tables
- ✅ Cached data loading for faster performance

## 📝 Notes

- The dataset may contain missing values which are handled during preprocessing
- Some computationally expensive operations (SVM, Hierarchical Clustering, LOO-CV) use sampled data for faster execution
- All models use random_state=42 for reproducibility
- The binary classification threshold is set at 1.0M global sales (Hit ≥ 1.0M, Flop < 1.0M)
- The Streamlit app caches data loading to improve performance on subsequent runs

## 🎓 Learning Outcomes

After completing this project, you will have:

- ✅ Implemented all major types of supervised learning (regression and classification)
- ✅ Applied unsupervised learning techniques (clustering and association rules)
- ✅ Used dimensionality reduction techniques (PCA)
- ✅ Built neural networks (MLP)
- ✅ Evaluated models using various metrics and cross-validation techniques
- ✅ Understood bias-variance trade-off
- ✅ Applied ensemble methods (Bagging, Boosting, Random Forest)

## 👤 Author

Created as part of Predictive Analytics coursework covering all six units of the syllabus.

## 📄 License

This project is for educational purposes.

---

**Note**: Make sure to have the `video_games_sales.csv` file in the same directory as the notebook before running.

