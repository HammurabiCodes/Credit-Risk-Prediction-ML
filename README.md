# Credit Risk Prediction Model

## Problem Statement
Banks need better ways to predict credit defaults. This project builds machine learning models to predict credit default risk using historical customer data.

## Dataset
- **Source:** UCI Machine Learning Repository (Default of Credit Card Clients)
- **Records:** 30,000 credit card holders
- **Features:** 24 variables (age, credit limit, payment history, etc.)
- **Target:** Binary (Default: Yes/No)

## Approach

### 1. Exploratory Data Analysis (EDA)
- Data quality assessment
- Feature distributions
- Correlation analysis
- Missing value handling

### 2. Feature Engineering
- Creating interaction features
- Feature scaling
- Dimensionality reduction

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
- Coming soon as model is developed

## Learning Outcomes
- Master machine learning workflow from data to deployment
- Understand credit risk modeling in fintech
- Apply Python + Scikit-learn to real-world problem
- Learn best practices in model evaluation

## Future Improvements
- Add deep learning models
- Implement feature selection techniques
- Create API for model predictions
- Deploy to production

## Contact
- **LinkedIn:** [linkedin.com/in/olivia-tamimi](https://www.linkedin.com/in/olivia-tamimi)
- **GitHub:** [github.com/HammurabiCodes](https://github.com/HammurabiCodes)

---
**Last Updated:** February 2026
**Status:** In Progress (Completing May 2026)
```

---