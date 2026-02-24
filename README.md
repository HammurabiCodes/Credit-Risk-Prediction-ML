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

### Week 1 (Feb 15-23, 2026) ✅ COMPLETED

✅ **Project Setup:**
- Initialized GitHub repository with professional documentation
- Dataset downloaded from Kaggle: 2,260,701 Lending Club loans (2007-2018)
- Verified 151 features including FICO scores, DTI ratios, payment history, loan characteristics

✅ **Dataset Overview:**
- **Total Loans:** 2,260,701
- **Time Period:** 2007-2018
- **Overall Default Rate:** 15.68%
- **Features:** 151 columns
- **Loan Amount Range:** $500 - $40,000 (Average: $15,047, Median: $12,900)
- **Average FICO Score:** 703
- **Average DTI Ratio:** 18.8%

---

**Analysis Completed:**

**1. FICO Score Impact**
- **FICO <660:** 30.88% default rate (HIGH RISK)
- **FICO 660-700:** 15.96% default rate
- **FICO 700-740:** 9.91% default rate
- **FICO 740-780:** 6.12% default rate
- **FICO >780:** 4.14% default rate (LOW RISK)

**Key Insight:** FICO scores below 660 show **7.5x higher default risk** compared to scores above 780. FICO will be the most important feature in our model.

---

**2. Loan Grade Performance**
| Grade | Default Rate | Risk Level |
|-------|--------------|------------|
| A | 3.59% | Lowest Risk |
| B | 8.66% | Low Risk |
| C | 14.36% | Moderate Risk |
| D | 20.35% | Elevated Risk |
| E | 28.28% | High Risk |
| F | 36.42% | Very High Risk |
| G | 40.01% | Highest Risk |

**Key Insight:** Grade G loans default at **11x the rate** of Grade A loans. Clear gradient validates Lending Club's risk assessment system.

**3. Debt-to-Income (DTI) Ratio** 💳
- **DTI 0-10%:** 9.92% default rate
- **DTI 10-20%:** 11.78% default rate
- **DTI 20-30%:** 14.80% default rate
- **DTI 30-40%:** 17.37% default rate
- **DTI >40%:** 9.74% default rate (anomaly - likely small sample size)

**Key Insight:** DTI ratios above 30% show **75% higher default risk** compared to DTI below 10%. Moderate predictor that strengthens when combined with other features.

**4. Loan Purpose Analysis** 🎯

**Highest Risk Purposes:**
- **Educational:** 20.75% default rate
- **Small Business:** 20.22% default rate
- **Renewable Energy:** 16.33% default rate

**Lowest Risk Purposes:**
- **Car:** 9.78% default rate
- **Credit Card:** 10.46% default rate
- **Home Improvement:** 11.31% default rate

**Key Insight:** Small business and educational loans are **2x riskier** than car loans. Loan purpose indicates borrower financial stability and should be included as categorical feature.

---
**5. Financial Crisis Impact (2007-2018)** 📉

**Default Rates by Year:**
- **2007:** 26.20% ⚠️ (Pre-crisis, early defaults)
- **2008:** 20.73% ⚠️ (Crisis peak)
- **2009:** 13.69% ⚠️ (Crisis recovery begins)
- **2010-2016:** 14-18% (Stabilization)
- **2017:** 10.63% (Improving economy)
- **2018:** 3.25% (Recent loans, less time to default)

**Key Insight:** 
- 2007-2008 loans show **8x higher default rates** than 2018 loans
- Crisis period demonstrates importance of macroeconomic factors
- More recent years have lower observed defaults (loans haven't matured yet)
- **Time-based features critical for model accuracy**

---
**6. Loan Amount Distribution** 💰
- **$0-5K:** Moderate default risk
- **$5K-10K:** Slightly lower risk
- **$10K-15K:** Average risk
- **$15K-20K:** Moderate risk
- **$20K-40K:** Elevated risk

**Key Insight:** Larger loans (>$20K) show moderately higher default rates. Loan amount interaction with income (debt-service coverage) will be important feature to engineer.

---

**7. Data Quality Assessment** 🔍
- **Member ID:** 100% missing (can drop)
- **Hardship columns:** 99.5% missing (hardship program data - will drop)
- **Core features:** Complete with <5% missing values
- **Action:** Will handle missing values in Week 2 feature engineering

---

**Business Implications:**

**Risk Segmentation Opportunity:**
- **Ultra-Low Risk:** FICO >740, Grade A-B, DTI <20% (default rate: ~4-6%)
- **Manageable Risk:** FICO 660-740, Grade C-D, DTI 20-30% (default rate: ~12-16%)
- **High Risk:** FICO <660, Grade E-G, DTI >30% (default rate: ~25-40%)

**Current lending may be too conservative on 660-700 FICO range with strong secondary signals** (low DTI, stable employment, home improvement purpose). Machine learning can identify these approval opportunities.

**Estimated Business Impact:**
- Potential to approve additional 50,000-100,000 creditworthy borrowers annually
- Estimated revenue impact: $2-5M from interest on approved loans
- Risk mitigation through better pricing (interest rates aligned with predicted default probability)

---

**Deliverables:**
- ✅ `notebooks/01_data_exploration.ipynb` - Complete exploratory analysis
- ✅ 6 professional visualizations saved in `results/` folder:
  - `default_by_fico.png`
  - `default_by_grade.png`
  - `default_by_dti.png`
  - `default_by_purpose.png`
  - `default_over_time.png`
  - `default_by_loan_amount.png`
- ✅ Comprehensive documentation of findings and insights

---
**Key Features Identified for Model Development:**

**Primary Predictors (Strong signals):**
- FICO score (strongest single predictor)
- Loan grade (validated risk assessment)
- DTI ratio (debt burden indicator)

**Secondary Predictors (Moderate signals):**
- Loan purpose (risk category indicator)
- Issue date / economic period (macro factors)
- Loan amount (size-based risk)

**Feature Engineering Opportunities:**
- FICO bins (categorical: <660, 660-700, 700-740, 740+)
- DTI categories (low/medium/high risk)
- Economic period indicators (crisis vs. stable periods)
- Loan purpose risk groupings (high/medium/low risk categories)
- Interaction features (FICO × DTI, loan_amount / annual_income)

---

**Next Steps - Week 2 (March 2, 2026):**
- Handle missing values systematically
- Feature engineering: Create derived features and interactions
- Encode categorical variables (one-hot encoding, target encoding)
- Create train/test split (80/20, stratified by target)
- Scale numerical features for modeling
- Prepare final dataset for baseline model development

📓 **[View Complete Analysis Notebook](notebooks/01_data_exploration.ipynb)**


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
**Last Updated:** February 24, 2026
**Status:** In Progress (Completing May 2026)
```