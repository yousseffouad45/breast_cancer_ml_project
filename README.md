# Breast Cancer Classification: Complete ML Pipeline

## 📋 Project Overview

This project implements a comprehensive machine learning pipeline for classifying breast cancer tumors as benign or malignant using 30 diagnostic features. The project includes:

✅ **Extensive Exploratory Data Analysis (EDA)**
- Distribution analysis for all 30 features
- Class comparison visualizations (benign vs malignant)
- Correlation analysis with heatmaps
- Feature importance rankings

✅ **Complete ML Workflow**
1. Problem Definition (Imbalanced data handling)
2. Data Exploration & Visualization
3. Preprocessing (standardization, feature scaling)
4. Feature Engineering & Dimensionality Reduction (PCA)
5. Model Training (Logistic Regression, Random Forest)
6. Evaluation (ROC-AUC, Confusion Matrix, Classification Report)

✅ **Interactive Streamlit Deployment**
- Real-time predictions on new data
- Dynamic EDA dashboard
- Model performance metrics
- Feature importance visualization

---

## 📁 Files Included

### 1. **breast_cancer_enhanced.ipynb**
   - Complete Jupyter notebook with ALL steps
   - Contains 40+ cells with extensive EDA
   - Includes data loading, exploration, preprocessing, modeling, and evaluation
   - Ready to run with `jupyter notebook breast_cancer_enhanced.ipynb`

### 2. **streamlit_app.py**
   - Web application for model deployment
   - 4 interactive pages: Predictions, EDA Dashboard, Model Performance, About
   - Beautiful UI with custom CSS styling

### 3. **README.md** (this file)
   - Setup instructions and usage guide

---

## 🚀 Quick Start

### Option A: Run the Jupyter Notebook

```bash
# 1. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# 2. Download the dataset
# Get breast-cancer.csv from UCI Machine Learning Repository:
# https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
# OR use scikit-learn's built-in dataset

# 3. Run Jupyter
jupyter notebook breast_cancer_enhanced.ipynb
```

### Option B: Run the Streamlit Web App

```bash
# 1. Install dependencies
pip install streamlit numpy pandas matplotlib seaborn scikit-learn

# 2. Run the app
streamlit run streamlit_app.py

# 3. Open browser to http://localhost:8501
```

---

## 📊 Dataset

**Source:** UCI Machine Learning Repository (Wisconsin Diagnostic Breast Cancer)

**Details:**
- **Samples:** 569 patients
- **Features:** 30 diagnostic measurements
- **Target:** Benign (0) or Malignant (1)
- **Class Distribution:** 
  - Benign: 357 (63%)
  - Malignant: 212 (37%)

**Feature Categories:**
Each tumor has 10 baseline measurements × 3 variants (mean, standard error, worst):
1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
6. Compactness
7. Concavity
8. Concave Points
9. Symmetry
10. Fractal Dimension

---

## 📈 Model Performance

### Best Model: Random Forest (100 trees)

| Metric | Score |
|--------|-------|
| Accuracy | 97.4% |
| Precision | 97.2% |
| Recall | 96.0% |
| F1-Score | 96.6% |
| ROC-AUC | 0.9901 |

### Comparison with Other Models

| Model | Features | Accuracy | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 30 | 96.5% | 0.9912 |
| Random Forest | 30 | 97.4% | 0.9901 |
| Logistic Regression + PCA-2 | 2 | 94.7% | 0.9649 |

### Top 5 Most Important Features
1. Worst concave points (13.7%)
2. Worst radius (12.8%)
3. Worst perimeter (11.9%)
4. Worst concavity (7.1%)
5. Mean concave points (6.5%)

---

## 🔍 Key Findings from EDA

### Findings
- ✓ **No missing values** - clean dataset
- ✓ **Features are right-skewed** - standardization is essential
- ✓ **High correlation between related features** (e.g., radius, perimeter, area)
- ✓ **Clear separation** between benign and malignant in key features
- ✓ **High correlation** with target (0.76 for worst concave points)

### Challenges
- ⚠️ **Class imbalance** (63% vs 37%) - handled via stratified split
- ⚠️ **Multicollinearity** - addressed with PCA analysis
- ⚠️ **High dimensionality** - 30 features for 569 samples

### Solutions Implemented
- ✅ Stratified train-test split (maintains class distribution)
- ✅ StandardScaler (zero-mean, unit-variance)
- ✅ PCA analysis (found 63% variance in 2 components)
- ✅ Random Forest (handles high dimensionality well)

---

## 🎯 Streamlit App Features

### 1. 🔮 Predictions Page
- Interactive sliders for all 30 features
- Real-time prediction with confidence scores
- Malignancy risk percentage
- Top 10 feature importance chart
- Color-coded predictions (Benign/Malignant)

### 2. 📊 EDA Dashboard
**Distributions Tab:**
- Individual feature distributions
- Box plots comparing classes

**Correlations Tab:**
- Heatmap of top N features
- Feature correlation with target
- Identify highly correlated features

**Class Comparison Tab:**
- Class distribution charts
- Summary statistics by class

### 3. 📈 Model Performance
- 5-fold cross-validation scores
- Training accuracy metrics
- Confusion matrix with interpretation
- Performance comparison

### 4. ℹ️ About Page
- Project overview
- Dataset information
- Feature descriptions
- Model architecture details
- Disclaimer and usage notes

---

## 💻 Installation & Setup

### System Requirements
- Python 3.7+
- 2GB RAM minimum
- Internet connection (for downloading dataset)

### Step-by-Step Setup

#### 1. Clone or Download Files
```bash
# Make sure you have all 3 files:
# - breast_cancer_enhanced.ipynb
# - streamlit_app.py
# - breast-cancer.csv (dataset)
```

#### 2. Create Virtual Environment (Recommended)
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy==1.23.5
pip install pandas==1.5.3
pip install scikit-learn==1.2.2
pip install matplotlib==3.7.1
pip install seaborn==0.12.2
pip install streamlit==1.25.0
pip install scipy==1.10.1
```

#### 4. Get the Dataset
Option A - Download from Kaggle:
```bash
# Visit: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
# Download breast_cancer.csv to your working directory
```

Option B - Use Jupyter notebook (loads via scikit-learn)

#### 5. Run Streamlit App
```bash
streamlit run streamlit_app.py
```

#### 6. Run Jupyter Notebook
```bash
jupyter notebook breast_cancer_enhanced.ipynb
```

---

## 📝 Notebook Contents

### Section 1: Problem Definition & Imports
- Import libraries
- Define SEED for reproducibility

### Section 2: Data Exploration (EDA)
- Load data and check shape
- Descriptive statistics
- Distribution analysis
- Outlier detection via boxplots
- Correlation analysis

### Section 3: Preprocessing
- Train-test split with stratification
- StandardScaler normalization

### Section 4: Feature Engineering
- PCA variance analysis
- Cumulative variance plot
- Component selection

### Section 5: Model Training
- Baseline: Logistic Regression (30 features)
- Random Forest (30 features)
- PCA + Logistic Regression (2 features)

### Section 6: Evaluation
- Classification reports
- Confusion matrices
- ROC curves
- Model comparison

### Section 7: Visualization
- 2D PCA projection
- Feature importance plots
- Performance metrics comparison

### Section 8: Deployment
- Save trained model
- Streamlit instructions

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'sklearn'"
**Solution:** Install scikit-learn
```bash
pip install scikit-learn
```

### Issue: "FileNotFoundError: breast-cancer.csv not found"
**Solution:** 
- Download from Kaggle: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- OR modify notebook to load from scikit-learn:
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

### Issue: Streamlit app won't start
**Solution:** Ensure all dependencies are installed
```bash
pip install --upgrade streamlit
streamlit run streamlit_app.py --logger.level=debug
```

### Issue: Jupyter kernel not found
**Solution:** Install and select correct kernel
```bash
python -m ipykernel install --user --name myenv
```

---

## 📚 Learning Outcomes

After completing this project, you will understand:

1. **End-to-end ML Pipeline**
   - Data loading → Preprocessing → Training → Evaluation → Deployment

2. **Exploratory Data Analysis (EDA)**
   - Statistical analysis
   - Distribution visualization
   - Correlation analysis
   - Outlier detection

3. **Feature Engineering**
   - Standardization/Normalization
   - Dimensionality Reduction (PCA)
   - Feature Selection
   - Multicollinearity handling

4. **Model Training**
   - Baseline models
   - Ensemble methods (Random Forest)
   - Hyperparameter tuning
   - Cross-validation

5. **Model Evaluation**
   - Classification metrics (Accuracy, Precision, Recall, F1)
   - ROC-AUC curves
   - Confusion matrices
   - Model comparison

6. **Web Deployment**
   - Streamlit framework
   - Interactive UI design
   - Real-time predictions
   - Dashboard creation

---

## 🔐 Disclaimer

**⚠️ IMPORTANT: Medical Use Disclaimer**

This model is for **educational and research purposes only**. It should NOT be used for:
- Actual medical diagnosis
- Treatment decisions
- Clinical judgment
- Any medical application without professional review

**Always consult with qualified healthcare professionals** for medical decisions.

---

## 📞 Support & Contact

For questions or issues:
1. Check the troubleshooting section above
2. Review the notebook comments
3. Check Streamlit documentation: https://docs.streamlit.io
4. Scikit-learn documentation: https://scikit-learn.org

---

## 📄 License

This project uses the UCI Machine Learning Repository's Breast Cancer Wisconsin dataset.
Attribution: Dr. William H. Wolberg, University of Wisconsin

---

## 🎓 Project Statistics

- **Total Cells:** 45+ in notebook
- **Code Lines:** 1,500+
- **Visualizations:** 20+
- **Models Trained:** 3
- **EDA Sections:** 8
- **Documentation:** Comprehensive

---

## 🏆 Best Practices Demonstrated

✅ Clear problem statement  
✅ Comprehensive EDA  
✅ Data preprocessing  
✅ Feature engineering  
✅ Multiple model comparison  
✅ Thorough evaluation  
✅ Interactive deployment  
✅ Code documentation  
✅ Error handling  
✅ Reproducible (SEED)  

---

**Created:** 2026  
**Version:** 1.0  
**Status:** Production Ready ✅
