"""
==================================================================
QUANTUM MUTUAL FUND - LOGISTIC REGRESSION FROM SCRATCH
STEP 2: Hypothesis Function, Maximum Likelihood Estimation, Log-Loss
==================================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

DATA_PATH = '/mnt/user-data/outputs/quantum_mutual_fund_leads.csv'


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------
# 2.1  HYPOTHESIS FUNCTION
# ------------------------------------------------------------------
def hypothesis(beta, x):
    """h_beta(x) = sigmoid(beta . x)"""
    return sigmoid(np.dot(beta, x))


print("="*70, "\n2.1 HYPOTHESIS FUNCTION\n", "="*70, sep="")
beta_demo = np.array([-4.2, 0.018, 1.4, 0.9])
x_demo = np.array([1, 60, 0.447, 0])
print(f"  h_beta(x) = {hypothesis(beta_demo, x_demo):.4f}")


# ------------------------------------------------------------------
# 2.2  LIKELIHOOD FUNCTION FOR A SINGLE EXAMPLE
# ------------------------------------------------------------------
def single_likelihood(y_i, p_i):
    return (p_i ** y_i) * ((1 - p_i) ** (1 - y_i))


print("\n" + "="*70)
print("2.2 SINGLE-EXAMPLE LIKELIHOOD")
print("="*70)
print(f"  y=1, p=0.80 -> Likelihood = {single_likelihood(1, 0.80):.4f}  (correct & confident)")
print(f"  y=0, p=0.80 -> Likelihood = {single_likelihood(0, 0.80):.4f}  (wrong & confident)")


# ------------------------------------------------------------------
# 2.3  TOTAL LIKELIHOOD (PRODUCT) AND UNDERFLOW DEMONSTRATION
# ------------------------------------------------------------------
toy_y = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
toy_p = np.array([0.81, 0.65, 0.22, 0.74, 0.31, 0.88, 0.19, 0.41, 0.69, 0.77])

total_L = np.prod(single_likelihood(toy_y, toy_p))
print("\n" + "="*70)
print("2.3 TOTAL LIKELIHOOD ACROSS 10 EXAMPLES (PRODUCT)")
print("="*70)
print(f"  Total Likelihood = {total_L:.10f}  (already tiny -- underflows with 800 rows)")


# ------------------------------------------------------------------
# 2.4  LOG-LIKELIHOOD (SUM FORM)
# ------------------------------------------------------------------
def log_likelihood(y, p):
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


ll = log_likelihood(toy_y, toy_p)
print("\n" + "="*70)
print("2.4 LOG-LIKELIHOOD (SUM FORM)")
print("="*70)
print(f"  Log-Likelihood = {ll:.4f}")
print(f"  exp(Log-Likelihood) = {np.exp(ll):.10f}  (matches the direct product above)")


# ------------------------------------------------------------------
# 2.5  LOG-LOSS / BINARY CROSS-ENTROPY COST FUNCTION
# ------------------------------------------------------------------
def log_loss_manual(y, p):
    m = len(y)
    return -(1/m) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


J = log_loss_manual(toy_y, toy_p)
print("\n" + "="*70)
print("2.5 LOG-LOSS  J(beta) = -average(Log-Likelihood)")
print("="*70)
print(f"  J(beta) for 10 toy examples = {J:.4f}")


# ------------------------------------------------------------------
# 2.6  VALIDATING MANUAL LOG-LOSS AGAINST sklearn.metrics.log_loss ON REAL DATA
# ------------------------------------------------------------------
print("\n" + "="*70)
print("2.6 VALIDATING AGAINST sklearn.metrics.log_loss (REAL 800-ROW DATASET)")
print("="*70)

df = pd.read_csv(DATA_PATH)
df_enc = pd.get_dummies(
    df.drop(columns=['customer_id']),
    columns=['gender', 'occupation', 'education', 'marital_status', 'city_tier', 'risk_appetite'],
    drop_first=True
)
X = df_enc.drop(columns=['invested'])
y = df_enc['invested']
X_scaled = StandardScaler().fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)
p_all = model.predict_proba(X_scaled)[:, 1]

manual_J = log_loss_manual(y.values, p_all)
sklearn_J = log_loss(y, p_all)

print(f"  Manual J(beta) formula        : {manual_J:.6f}")
print(f"  sklearn.metrics.log_loss()    : {sklearn_J:.6f}")
print(f"  Match: {np.isclose(manual_J, sklearn_J)}")


# ------------------------------------------------------------------
# 2.7  PROVING argmax(Log-Likelihood) == argmin(Log-Loss) ON REAL DATA
# ------------------------------------------------------------------
print("\n" + "="*70)
print("2.7 GRID SEARCH: PROVING MLE AND LOG-LOSS GIVE THE SAME OPTIMAL BETA")
print("="*70)

x_email = df['email_open_rate'].values
y_real = df['invested'].values
beta0_fixed = -1.0
beta_candidates = np.linspace(-2, 5, 400)

ll_vals, loss_vals = [], []
for b in beta_candidates:
    z = beta0_fixed + b * x_email
    p = np.clip(sigmoid(z), 1e-10, 1 - 1e-10)
    ll_vals.append(log_likelihood(y_real, p))
    loss_vals.append(log_loss_manual(y_real, p))

best_by_ll = beta_candidates[np.argmax(ll_vals)]
best_by_loss = beta_candidates[np.argmin(loss_vals)]

print(f"  Beta maximizing Log-Likelihood : {best_by_ll:.4f}")
print(f"  Beta minimizing Log-Loss       : {best_by_loss:.4f}")
print(f"  Identical: {np.isclose(best_by_ll, best_by_loss)}")

print("\n" + "="*70)
print("STEP 2 COMPLETE")
print("="*70)
