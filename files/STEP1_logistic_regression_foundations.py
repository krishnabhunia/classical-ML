"""
==================================================================
QUANTUM MUTUAL FUND - LOGISTIC REGRESSION FROM SCRATCH
STEP 1: Foundations - Classification, Sigmoid, Logit, Odds
==================================================================
Run this file top to bottom. Each section is self-contained and
prints its own labeled output.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA_PATH = '/mnt/user-data/outputs/quantum_mutual_fund_leads.csv'


# ------------------------------------------------------------------
# 1.1  THE SIGMOID FUNCTION
# ------------------------------------------------------------------
def sigmoid(z):
    """sigma(z) = 1 / (1 + e^-z) -- squashes any real number into (0,1)"""
    return 1 / (1 + np.exp(-z))


def logit(p):
    """logit(p) = ln(p / (1-p)) -- the inverse of sigmoid"""
    return np.log(p / (1 - p))


print("="*70, "\n1.1 SIGMOID FUNCTION\n", "="*70, sep="")
for z in [-5, -1, 0, 1, 5]:
    print(f"  sigmoid({z:>3}) = {sigmoid(z):.4f}")


# ------------------------------------------------------------------
# 1.2  WHY LINEAR REGRESSION FAILS (sklearn.linear_model.LinearRegression)
# ------------------------------------------------------------------
print("\n" + "="*70)
print("1.2 WHY LINEAR REGRESSION FAILS FOR CLASSIFICATION (sklearn)")
print("="*70)

np.random.seed(1)
x_toy = np.concatenate([np.random.normal(2, 1, 20), np.random.normal(9, 1, 20)]).reshape(-1, 1)
y_toy = np.concatenate([np.zeros(20), np.ones(20)])

lin_reg = LinearRegression().fit(x_toy, y_toy)
preds = lin_reg.predict(np.array([[15], [-5]]))
print(f"  LinearRegression predicts {preds[0]:.3f} for x=15   <- invalid probability (>1)")
print(f"  LinearRegression predicts {preds[1]:.3f} for x=-5   <- invalid probability (<0)")


# ------------------------------------------------------------------
# 1.3  ODDS, LOG-ODDS, AND THE SIGMOID/LOGIT INVERSE RELATIONSHIP
# ------------------------------------------------------------------
print("\n" + "="*70)
print("1.3 ODDS AND LOG-ODDS (LOGIT)")
print("="*70)

p = 0.75
print(f"  p = {p}")
print(f"  odds(p)  = p/(1-p)         = {p/(1-p):.4f}")
print(f"  logit(p) = ln(odds)        = {logit(p):.4f}")
print(f"  sigmoid(logit(p))          = {sigmoid(logit(p)):.4f}  <- proves they are inverses")


# ------------------------------------------------------------------
# 1.4  FITTING REAL sklearn LogisticRegression ON QUANTUM MUTUAL FUND DATA
# ------------------------------------------------------------------
print("\n" + "="*70)
print("1.4 FITTING sklearn.linear_model.LogisticRegression ON REAL DATA")
print("="*70)

df = pd.read_csv(DATA_PATH)

df_enc = pd.get_dummies(
    df.drop(columns=['customer_id']),
    columns=['gender', 'occupation', 'education', 'marital_status', 'city_tier', 'risk_appetite'],
    drop_first=True
)

X = df_enc.drop(columns=['invested'])
y = df_enc['invested']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

print(f"  Intercept (beta_0) learned by sklearn : {model.intercept_[0]:.4f}")
print(f"  Number of feature weights learned     : {len(model.coef_[0])}")
print(f"  Training accuracy                     : {model.score(X_scaled, y):.4f}")

# Show top 5 strongest coefficients
coef_table = pd.DataFrame({'feature': X.columns, 'beta': model.coef_[0]})
coef_table = coef_table.reindex(coef_table.beta.abs().sort_values(ascending=False).index)
print("\n  Top 5 strongest learned weights:")
print(coef_table.head(5).to_string(index=False))


# ------------------------------------------------------------------
# 1.5  ODDS RATIO INTERPRETATION OF A REAL FITTED COEFFICIENT
# ------------------------------------------------------------------
print("\n" + "="*70)
print("1.5 ODDS RATIO INTERPRETATION (using the real fitted past_sip_holder weight)")
print("="*70)

beta_sip = coef_table.loc[coef_table['feature'] == 'past_sip_holder', 'beta'].values[0]
odds_ratio = np.exp(beta_sip)
print(f"  beta (past_sip_holder)  = {beta_sip:.4f}")
print(f"  Odds Ratio = e^beta     = {odds_ratio:.4f}")
print(f"  --> A past SIP holder has {odds_ratio:.2f}x the odds of investing again,")
print(f"      holding all other features constant.")


# ------------------------------------------------------------------
# 1.6  PREDICTING PROBABILITY FOR ONE REAL CUSTOMER (using the trained sklearn model)
# ------------------------------------------------------------------
print("\n" + "="*70)
print("1.6 PREDICTING A REAL CUSTOMER'S PROBABILITY (trained sklearn model)")
print("="*70)

sample_customer = X_scaled[0].reshape(1, -1)
predicted_prob = model.predict_proba(sample_customer)[0][1]
actual_label = y.iloc[0]
print(f"  Customer: {df.iloc[0]['customer_id']}")
print(f"  Model's predicted probability of investing : {predicted_prob:.4f}")
print(f"  Actual outcome                              : {actual_label}")

print("\n" + "="*70)
print("STEP 1 COMPLETE")
print("="*70)
