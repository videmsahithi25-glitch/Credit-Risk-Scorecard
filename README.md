# Credit Risk Scorecard

An end-to-end credit risk scorecard built using the Home Credit Default Risk 
dataset from Kaggle. The goal is to predict the probability of loan default 
for each customer using Weight of Evidence (WoE) encoding and Information Value 
(IV) for feature selection, trained on a Logistic Regression model. The final 
output is a scaled scorecard with point values evaluated using KS Statistic, 
Gini Coefficient, ROC curve and Calibration plots.

## Dataset
Home Credit Default Risk — Kaggle (7 tables)

| Table | Description |
|-------|-------------|
| application_train | Main table with customer demographics and loan details |
| bureau | Credit history from other financial institutions |
| bureau_balance | Monthly balance of bureau credits |
| previous_application | Previous loan applications at Home Credit |
| POS_CASH_balance | Monthly balance of previous POS and cash loans |
| installments_payments | Repayment history of previous loans |
| credit_card_balance | Monthly balance of previous credit card loans |


## Project Pipeline

### 1. Data Pipeline
- All 7 tables aggregated and merged into application_train using SK_ID_CURR
- Left join used throughout to retain all customers
- Each supplementary table aggregated to one row per customer before merging

### 2. Exploratory Data Analysis
- Analysed distributions, missing values and class imbalance
- Dataset shape: 44,651 rows × 122 columns
- Target variable: 91.9% Non-Default, 8.1% Default

### 3. Data Preprocessing
- Missing values reduced from 1,329,598 to 0 after treatment
- Numerical columns filled with median values to avoid data leakage
- Binary missing flags created to preserve missingness as a signal
- Rare label encoding applied for categories below 1% frequency
- 5 new ratio features engineered: CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, 
  CREDIT_GOODS_RATIO, EMPLOYED_TO_AGE_RATIO, INCOME_PER_PERSON

### 4. WoE Binning & IV Feature Selection
- OptimalBinning used for binning with monotonicity enforcement
- IV threshold of 0.02 applied
- Features reduced from 267 to 62

### 5. Model Building
- Algorithm: Logistic Regression with L1 regularization
- Solver: liblinear | C: 0.1 | max_iter: 1000
- Train/Test split: 80/20 with stratified sampling

### 6. Scorecard Scaling
- PDO = 20 | Base Score = 600 | Base Odds = 50
- Final scorecard assigns credit score to each customer based on default probability


## Model Performance

| Metric | Value |
|--------|-------|
| AUC | 0.7549 |
| Gini Coefficient | 0.5099 |
| KS Statistic | 0.3822 |
| Accuracy | 0.91 |
| Precision (Class 1) | 0.39 |
| Recall (Class 1) | 0.11 |
| F1 Score (Class 1) | 0.18 |



## Score Band Distribution

| Score Band | Total Customers | Default Rate |
|------------|----------------|--------------|
| <500 | 535 | 45.2% |
| 500-520 | 2332 | 30.5% |
| 520-540 | 7081 | 18.2% |
| 540-560 | 13768 | 10.1% |
| 560-580 | 17054 | 5.1% |
| 580-600 | 13405 | 2.6% |
| 600-620 | 5850 | 1.6% |
| 620-640 | 1336 | 1.2% |
| 640+ | 142 | 0.7% |



## Notebooks
- `CreditRisk_DataPipeline.ipynb` — Data merging and aggregation
- `ApplicationTrain_EDA_.ipynb` — Exploratory data analysis
- `CreditScorecard_Modelling.ipynb` — WoE, modelling and scorecard# Credit-Risk-Scorecard
End-to-end credit risk scorecard using WoE, IV and Logistic Regression
