# Credit Risk Prediction Model

## Problem Statement
Banks need better ways to predict credit defaults. This project builds machine learning models to predict credit default risk using historical customer data.

As a risk management professional with 10+ years in fintech lending, I've seen firsthand how traditional credit scoring misses creditworthy applications while exposing lenders to unexpected defaults. This project combines my domain expertise with machine learning to build more accurate, explainable credit risk models. 

## Project Goals
**Technical:** Build interpretable ML models achieving >0. AUS-ROC credit default prediction

**Business:** Reduce falce decline rate by 15% while maintaining risk appetite, enabling $2-3M additional revenue from approved creditworthy applicants

**Learning:** Master end-to-end ML workflow from EDA to prodcution deployment in fintech context

## Current Focus (Week of Feb 15, 2026)
- Loading and exploring the dataset
- Understanding feature distributions and target class balance
- Identifying missing values and data quality issues
- Creating Initial visualizations of default patterns

**Next Week:** Feature engineering and baseline model development

## Dataset
- **Source:** UCI Machine Learning Repository (Default of Credit Card Clients)
- **Records:** 30,000 credit card holders
- **Features:** 24 variables (age, credit limit, payment history, etc.)
- **Target:** Binary (Default: Yes/No)
- **Business Context:** Predicting credit card default to optimize credit line decision and reduce portfolio risk for financial institutions.

## Approach

### 1. Exploratory Data Analysis (EDA)
- Data quality assessment
- Feature distributions
- Correlation analysis
- Missing value handling
- Identify credit behavior patterns based on risk management domain expertise 
- Analyze payment-to-limit ratios and utilization trends (key default indicators)

### 2. Feature Engineering
- Creating interaction features
- Feature scaling
- Dimensionality reduction
- Creating debt-service coverage proxies
- Engineer behavioral trend features (payment consistency, credit seeking behavior)


### 3. Model Selection & Training
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- Gradient Boosting

### 4. Evaluation
- ROC-AUC Score
- Confusion Matrix
- Precision / Recall / F1-Score
- Feature Importance Analysis

## Tech Stack
- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
- **Matplotlib & Seaborn** - Visualization
- **Jupyter Notebook** - Interactive analysis

## Project Structure
```
Credit-Risk-Prediction-ML/
├── data/                          # Datasets
├── notebooks/                     # Jupyter notebooks
│   └── Credit_Risk_Analysis.ipynb
├── src/                           # Python modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── model.py
├── results/                       # Outputs
│   ├── visualizations/
│   └── models/
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

### 1. Clone Repository
```bash
git clone https://github.com/HammurabiCodes/Credit-Risk-Prediction-ML.git
cd Credit-Risk-Prediction-ML
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Analysis
```bash
jupyter notebook
```

## Key Findings (To be updated with results)
Week 1 (Feb 15 2026):
- ✅ Project initialized, dataset loaded
- ✅ Identified 30,000 records with 24 features
- 🔄 Next: Exploratory analysis and missing value assessment 

[Will update weekly discoveries]

## Learning Outcomes
- Master machine learning workflow from data to deployment
- Understand credit risk modeling in fintech
- Apply Python + Scikit-learn to real-world problem
- Learn best practices in model evaluation

## Future Improvements
- Implementing SHAP values for model explainability (regulatory requirement in lending)
- Adding ensemble stacking (combining XGBoost + Logistic Regression)
- Building interactive streamlit dashboard for credit officers 
- Integrate with real-time scoring API (production deployment)
- A/B testing framework for champion/challenger model comparison 

## Contact
- **LinkedIn:** [linkedin.com/in/olivia-tamimi](https://www.linkedin.com/in/olivia-tamimi)
- **GitHub:** [github.com/HammurabiCodes](https://github.com/HammurabiCodes)

---
**Last Updated:** February 2026
**Status:** In Progress (Completing May 2026)
```

---