# Credit Risk Prediction Model

## Problem Statement
Peer-to-peer lending platforms like Lending Club, Upstart, and Prosper revolutionized consumer credit by connecting borrowers directly with investors. However, accurate default prediction is critical for platform success:

**Key Challenges:**
1. **Credit Access:** Traditional FICO-only models reject creditworthy borrowers with thin credit files
2. **Risk Pricing:** Interest rates must accurately reflect default probability to protect investor returns
3. **Portfolio Performance:** High default rates erode investor confidence and platform viability

**The Opportunity:**
As a risk management professional with 10+ years in fintech lending, I've seen how traditional credit scoring models miss nuanced risk signals in alternative data. This project applies machine learning to predict loan defaults using rich behavioral and financial features beyond traditional credit scores.

**Goal:** Build interpretable ML models that reduce false decline rates by 15% while maintaining portfolio risk within acceptable thresholds, enabling $2-3M in additional revenue from approved creditworthy applicants.

## Project Goals
**Technical:** Build interpretable ML models achieving >0.85 AUC-ROC for credit default prediction

**Business:** Reduce false decline rate by 15% while maintaining risk appetite, enabling an estimated $2-3M in additional revenue from approved creditworthy applicants

**Learning:** Master end-to-end ML workflow from EDA to production deployment in fintech context

## Current Focus (Week of Feb 15, 2026)
- Loading and exploring the dataset
- Understanding feature distributions and target class balance
- Identifying missing values and data quality issues
- Creating Initial visualizations of default patterns

**Next week:** Feature engineering and baseline model development.

## Dataset
- **Source:** Lending Club Loan Data (2007-2018)
- **Platform:** Kaggle / Lending Club public data
- **Link:** https://www.kaggle.com/datasets/wordsforthewise/lending-club
- **Records:** 2.9M+ personal loans from US peer-to-peer lending marketplace
- **Features:** 150+ variables including:
    - **Credit Metrics:** FICO score, credit history length, delinquencies, public records
    - **Loan Characteristics:** Amount, term (36/60 months), interest rate, grade (A-G), purpose
    - **Borrower Profile:** Annual income, employment length, home ownership, debt-to-income ratio
    - **Behavioral Data:** Payment history, credit utilization, number of open accounts
- **Target:** Loan status (Fully Paid, Charged Off, Default, Late, Current)
- **Time Period:** 2007-2018 (includes pre/post financial crisis data)
- **Geographic Coverage:** Across all 50 US states
- **Business Context:** Predicting loan default probability for peer-to-peer lending platform to optimize credit decisions, price risk appropriately, and protect investor returns while expanding access to creditworthy borrowers.

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

## 📈 Key Findings (Updated Weekly)

### Week 1 (Feb 15, 2026)

✅ **Project Setup:**
- Initialized GitHub repository with professional documentation
- Dataset downloaded from Kaggle: 2.9M+ Lending Club loans (2007-2018)
- Verified 150+ features including FICO scores, DTI ratios, payment history

✅ **Dataset Overview:**
- Loan amounts: $500-$40,000
- Loan terms: 36 or 60 months
- Credit grades: A (best) through G (subprime)
- Time period: 2007-2018 (includes financial crisis impact)

🔄 **Upcoming Analysis (Sunday Feb 15):**
- Calculate default rates overall and by loan grade
- Analyze FICO score distributions and correlation with defaults
- Examine DTI ratios and their impact on credit risk
- Identify missing values and data quality issues
- Create 6-8 professional visualizations

**Next Week:** Feature engineering and baseline model development

[Detailed findings will be added after completing exploratory analysis]

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
- **LinkedIn:** [linkedin.com/in/olivia-tamimi](https://linkedin.com/in/olivia-tamimi)
- **GitHub:** [github.com/HammurabiCodes](https://github.com/HammurabiCodes)

---
**Last Updated:** February 2026
**Status:** In Progress (Completing May 2026)
```