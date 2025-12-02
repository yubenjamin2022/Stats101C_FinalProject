import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier as xgb
from sklearn.metrics import roc_auc_score, log_loss
from lightgbm import LGBMClassifier


train = pd.read_csv("aluminum_coldRoll_train.csv")
test = pd.read_csv("aluminum_coldRoll_testNoY.csv")
train = train.drop(columns=["ID"])
test = test.drop(columns=["ID"])
# no nulls/NAs

target = "y_passXtremeDurability"
features = [c for c in train.columns if c != target]

# -----------------------------------------------------
# 5. Label encode ALL categorical (object) columns
# -----------------------------------------------------
cat_cols = train.select_dtypes(include="object").columns

for col in cat_cols:
    all_vals = pd.concat([train[col], test[col]]).astype(str)
    mapping = {v: i for i, v in enumerate(all_vals.unique())}

    train[col] = train[col].astype(str).map(mapping)
    test[col]  = test[col].astype(str).map(mapping)

X_train, X_val, y_train, y_val = train_test_split(
    train[features], train[target],
    test_size=0.2,
    random_state=42
)

# -----------------------------------------------------
# 7. Convert to XGBoost DMatrix
# -----------------------------------------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)
dtest  = xgb.DMatrix(test[features])

# -----------------------------------------------------
# 8. XGBoost parameters (good defaults)
# -----------------------------------------------------
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 2.0,
    "alpha": 0.1,
}

# -----------------------------------------------------
# 9. Train model
# -----------------------------------------------------
watchlist = [(dtrain, "train"), (dval, "val")]

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1500,
    evals=watchlist,
    early_stopping_rounds=50,
    verbose_eval=100
)

# -----------------------------------------------------
# 10. Validate model
# -----------------------------------------------------
val_pred = model.predict(dval)
auc = roc_auc_score(y_val, val_pred)
print("Validation AUC:", auc)

# -----------------------------------------------------
# 11. Predict on test
# -----------------------------------------------------
test_pred = model.predict(dtest)

# Ensure probabilities are in (0,1)
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

# -----------------------------------------------------
# 12. Create submission file
# -----------------------------------------------------
submission = pd.DataFrame()
submission["ID"] = test["ID"]
submission["y_passXtremeDurability"] = test_pred

submission.to_csv("xgboost_submission.csv", index=False)
