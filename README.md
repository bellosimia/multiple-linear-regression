# Marketing Budget Allocation — Multiple Linear Regression Analysis

## Dataset

**File:** `marketing_sales_data.csv`  
**Rows:** 572 | **Columns:** 5

|Column      |Type                                     |Description                   |
|------------|-----------------------------------------|------------------------------|
|TV          |Categorical (Low / Medium / High)        |TV advertising spend level    |
|Radio       |Continuous (float)                       |Radio advertising spend       |
|Social Media|Continuous (float)                       |Social media advertising spend|
|Influencer  |Categorical (Nano / Micro / Macro / Mega)|Influencer partnership tier   |
|Sales       |Continuous (float)                       |Target variable — units sold  |

## Analysis Goals

1. Perform exploratory data analysis on all variables
1. Check for multicollinearity using correlation matrices and Variance Inflation Factor (VIF)
1. Build a Multiple Linear Regression model using `statsmodels`-equivalent OLS via `sklearn` + `scipy`
1. Validate all four core OLS assumptions (linearity, normality, homoscedasticity, independence)
1. Interpret coefficients in business terms
1. Deliver a prioritised, evidence-based budget recommendation

## Key Results

|Metric              |Full Model (TV + Radio + Social Media)|Reduced Model (TV + Radio)|
|--------------------|--------------------------------------|--------------------------|
|R²                  |0.9039                                |0.9039                    |
|Adjusted R²         |0.9034                                |0.9036                    |
|Social Media p-value|0.8147 (not significant)              |— (excluded)              |

**Final Model Equation:**

```
Sales = 64.93 + 77.32 × TV_level + 2.96 × Radio
```

Where TV_level is encoded as: Low = 0, Medium = 1, High = 2.

## Environment Setup

```bash
pip install pandas numpy matplotlib scikit-learn scipy jupyter
```

Then open the notebook:

```bash
jupyter notebook multiple_regression_analysis.ipynb
```

## Files

```
├── multiple_regression_analysis.ipynb  # Main analysis notebook
├── marketing_sales_data.csv            # Dataset
├── README.md                           # This file
├── eda_distributions.png               # EDA plots (generated on run)
├── correlation_heatmap.png             # Correlation heatmap (generated on run)
├── diagnostic_plots.png                # Assumption diagnostics (generated on run)
└── model_performance.png               # Actual vs Predicted + coefficients (generated on run)
```

## Notebook Structure

1. **Setup and Data Loading** — imports, load CSV, initial inspection
1. **Exploratory Data Analysis** — distributions, mean Sales by channel level, scatter plots
1. **Multicollinearity Check** — correlation matrix heatmap + manual VIF computation
1. **Full OLS Model** — all three numeric predictors, full statistical output with p-values
1. **Reduced Model** — TV + Radio only, justified by Social Media p = 0.81
1. **Assumption Diagnostics** — four diagnostic plots + Shapiro-Wilk, Breusch-Pagan, Durbin-Watson
1. **Coefficient Interpretation** — business-language explanation of each coefficient
1. **Actual vs Predicted & Coefficient Visualisation** — performance charts
1. **Business Recommendation** — prioritised budget allocation guidance with limitations

## Statistical Methods

- **Multicollinearity:** Pearson correlation matrix + Variance Inflation Factor (VIF = 1/(1−R²) from auxiliary regressions)
- **Regression:** Ordinary Least Squares via `sklearn.linear_model.LinearRegression`
- **Inference:** Manual computation of standard errors, t-statistics, and p-values using the analytical OLS formula: SE(β) = sqrt(diag((X’X)⁻¹ · MSE))
- **Normality test:** Shapiro-Wilk on 200-observation sample
- **Homoscedasticity:** Breusch-Pagan test (n·R² from regressing squared residuals on X)
- **Autocorrelation:** Durbin-Watson statistic