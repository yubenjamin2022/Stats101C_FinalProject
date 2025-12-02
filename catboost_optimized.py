import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# -----------------------------
# Load data
# -----------------------------
train = pd.read_csv("aluminum_coldRoll_train.csv")
test = pd.read_csv("aluminum_coldRoll_testNoY.csv")

target = "y_passXtremeDurability"
ID_col = "ID"

test_IDs = test[ID_col]

train = train.drop(columns=[ID_col])
test = test.drop(columns=[ID_col])

cat_cols = list(train.select_dtypes(include=["object"]).columns)

X = train.drop(columns=[target])
y = train[target]

print(f"Train shape: {X.shape}")
print(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Optimized CatBoost Model
# -----------------------------
print("\nTraining Optimized CatBoost...")

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
val_pool = Pool(X_val, y_val, cat_features=cat_cols)
test_pool = Pool(test, cat_features=cat_cols)

model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="Logloss",
    learning_rate=0.04,           # Slightly slower for better convergence
    depth=9,                      # Deeper trees
    l2_leaf_reg=2,               # Less regularization
    min_data_in_leaf=15,         # More flexible leaves
    max_bin=254,                 # Maximum granularity
    bagging_temperature=0.3,     # Better generalization
    random_strength=0.3,         # Add randomness
    iterations=5000,             # More iterations
    random_state=42,
    od_type="Iter",
    od_wait=100,                 # Very patient early stopping
    verbose=100,
    thread_count=-1
)

model.fit(train_pool, eval_set=val_pool)

# Validation score
val_pred = model.predict_proba(val_pool)[:, 1]
val_score = log_loss(y_val, val_pred)
print(f"\nValidation LogLoss: {val_score:.6f}")

# Test predictions
test_pred = model.predict_proba(test_pool)[:, 1]

# Submission
submission = pd.DataFrame({
    "ID": test_IDs,
    "y_passXtremeDurability": test_pred
})

submission.to_csv("catboost_optimized.csv", index=False)
print(f"\nSaved catboost_optimized.csv")
print(f"Target: 0.424000")
print(f"Current: {val_score:.6f}")
if val_score < 0.424:
    print("✓ TARGET ACHIEVED!")
else:
    print(f"Gap: {val_score - 0.424:.6f}")