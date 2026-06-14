# Credit Card Default Prediction

## Question
Can we predict whether a credit card customer will default on their payment next month, using their demographic information and payment history?

## Dataset
- **Source:** UCI Machine Learning Repository — Credit Card Default Dataset
- **Size:** 30,000 real customers from a Taiwanese bank (2005)
- **Features:** Age, credit limit, education, marriage status, and 6 months of payment history and bill amounts
- **Target variable:** Whether the customer defaulted the following month (1 = yes, 0 = no)

## Methods & Statistical Concepts Used
1. **Distribution Analysis** — Explored credit limit distributions and default rates; found ~22% of customers defaulted (class imbalance)
2. **Correlation Analysis** — Built a heatmap across all 25 variables; payment history columns showed the strongest relationship with default
3. **Hypothesis Testing** — Two-sample t-test confirmed defaulters have significantly lower credit limits (p < 0.05)
4. **Logistic Regression** — Trained a binary classification model on 80% of the data to predict default probability
5. **Classification Metrics** — Evaluated model using ROC curve, AUC score, confusion matrix, and precision/recall

## Key Findings
- Customers who defaulted had significantly lower credit limits than those who didn't
- The most recent payment status (PAY_0) was the strongest predictor of default — recent behavior predicts future behavior
- The model achieved an **AUC of 0.721**, meaning it correctly identifies a defaulter over a non-defaulter 72% of the time
- Bill amounts had surprisingly low predictive power compared to payment status

## Tools Used
- Python (pandas, numpy, matplotlib, seaborn, scikit-learn, scipy)
- Jupyter Notebook
- VS Code

## Next Steps
- Test a Random Forest or XGBoost model to improve AUC beyond 0.721
- Add more features like debt-to-income ratio if available

