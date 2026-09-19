# Paisabazaar Bank Credit Risk Analytics

## The Problem
Banks and lending platforms need to know *before* approving a loan whether 
a customer is likely to default. Paisabazaar's customer credit dataset 
gives us a chance to build that early-warning system — classifying 
customers into Good / Standard / Poor credit risk tiers based on their 
financial behavior, not just a single credit score number.

## The Dataset
- ~100,000 customer records, 28 features covering income, debt-to-income 
  ratio, number of bank accounts, delayed payments, credit mix, and more
- Target: 3-class credit risk tier (Good / Standard / Poor)
- Classes were imbalanced — most customers fall in "Standard," with fewer 
  in the extreme "Poor" tier (which is, inconveniently, the group that 
  actually matters most for risk decisions)

## Approach
1. **EDA & cleaning** — checked correlation and multicollinearity across 
   financial features to avoid redundant/misleading predictors
2. **Feature engineering** — narrowed to a clean 12.5K-customer feature 
   set after removing noisy/duplicate signals
3. **Modeling** — compared Logistic Regression, Random Forest, and 
   XGBoost (tuned with Optuna), using class-weighting and sample-weighting 
   to counter the imbalance
4. **Evaluation** — prioritized **Recall** over Accuracy, for reasons 
   explained below

## Results (Optuna-Tuned XGBoost)

| Class    | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| Good     | 0.38      | 0.67   | 0.49     | 483     |
| Poor     | 0.58      | 0.70   | 0.63     | 720     |
| Standard | 0.72      | 0.43   | 0.54     | 1297    |
| **Macro Avg** | 0.56 | 0.60  | 0.55     | 2500    |

**Overall Accuracy: 55%** — deliberately not the headline metric (see below).

Best hyperparameters (via Optuna, 50+ trials): `n_estimators=344`, 
`max_depth=7`, `learning_rate=0.038`, `subsample=0.85`, 
`colsample_bytree=0.50`, `gamma=0.91`, `min_child_weight=2`.

## Reading the Trade-off
The model trades Standard-class recall for Poor-class recall — it catches 
70% of genuinely risky ("Poor") customers, at the cost of some 
misclassification in the safer, majority "Standard" group. In a lending 
context, missing a risky customer (false negative) is far costlier than 
over-flagging a safe one (false positive), so this trade-off is intentional, 
not a shortcoming.

The clearest remaining weakness is **Good-class precision (0.38)** — 
roughly 6 in 10 customers predicted "Good" are actually borderline 
"Standard" cases. This suggests the Good/Standard boundary needs either 
class-specific threshold tuning or additional features that separate 
consistently strong financial behavior from average behavior.
## What I'd Do Next
- Threshold tuning: shifting the classification threshold on the "Poor" 
  class probability to push recall higher (at a controlled precision cost)
- Try SMOTE/targeted oversampling on the minority risk class instead of 
  relying only on class weights
- Feature engineering pass: interaction terms between debt-to-income and 
  payment-delay count, which showed up as strong individual signals

## Tech Stack
Python · pandas · scikit-learn · XGBoost · Optuna · Matplotlib/Seaborn

## Repo Structure
`notebook.ipynb` — full analysis · `data/` — dataset · `README.md` — this file