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

# Save ID for later
test_IDs = test[ID_col]

# Drop ID from features
train = train.drop(columns=[ID_col])
test = test.drop(columns=[ID_col])

# -----------------------------
# Identify categorical columns
# -----------------------------
cat_cols = list(train.select_dtypes(include=["object"]).columns)

# -----------------------------
# Train/validation split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    train.drop(columns=[target]),
    train[target],
    test_size=0.2,
    random_state=42
)

# -----------------------------
# CatBoost pool
# -----------------------------
train_pool = Pool(X_train, y_train, cat_features=cat_cols)
val_pool   = Pool(X_val,   y_val,   cat_features=cat_cols)
test_pool  = Pool(test,            cat_features=cat_cols)

# -----------------------------
# CatBoost model
# -----------------------------
model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="Logloss",
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=5,
    iterations=4000,
    random_state=42,
    od_type="Iter",
    od_wait=80,
    verbose=200
)

model.fit(train_pool, eval_set=val_pool)

# -----------------------------
# Validation logloss
# -----------------------------
val_pred = model.predict_proba(val_pool)[:,1]
print("Validation LogLoss:", log_loss(y_val, val_pred))

# -----------------------------
# Predict on test
# -----------------------------
test_pred = model.predict_proba(test_pool)[:,1]

# -----------------------------
# Submission
# -----------------------------
submission = pd.DataFrame({
    "ID": test_IDs,
    "y_passXtremeDurability": test_pred
})

submission.to_csv("catboost_submission.csv", index=False)
print("Saved catboost_submission.csv")
