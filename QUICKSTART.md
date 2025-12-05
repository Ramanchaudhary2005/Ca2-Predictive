# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Streamlit App
```bash
streamlit run app.py
```

### Step 3: Open Browser
The app will automatically open at `http://localhost:8501`

## 📱 Using the Web Application

1. **Navigate** using the sidebar menu
2. **Select** a model or analysis type
3. **Click** "Run Model" or "Run Analysis" button
4. **View** results, metrics, and visualizations

## 🎯 Key Features

- **Interactive Interface**: Easy-to-use web interface
- **Real-time Results**: See model performance instantly
- **Visualizations**: Charts, graphs, and heatmaps
- **Model Comparison**: Compare multiple models side-by-side

## 📊 Available Pages

| Page | Description |
|------|-------------|
| Dataset Overview | View dataset statistics and preview |
| Data Preprocessing | See cleaning steps and feature engineering |
| Regression Models | Run regression algorithms |
| Classification Models | Test classification algorithms |
| Clustering | Perform clustering analysis |
| Neural Networks | Train MLP models |
| Ensemble Methods | Compare ensemble techniques |
| Model Comparison | Side-by-side model comparison |

## 💡 Tips

- Data is cached for faster loading
- Some models may take a few seconds to train
- Use the Model Comparison page to see all results at once
- Adjust parameters using sliders and selectboxes

## 🐛 Troubleshooting

**Issue**: Streamlit not found
```bash
pip install streamlit
```

**Issue**: Module not found errors
```bash
# Make sure you're in the project directory
# Check that all files are present
ls -la
```

**Issue**: Dataset not found
- Ensure `video_games_sales.csv` is in the same directory as `app.py`

## 📚 For More Details

See `README.md` for comprehensive documentation.

