import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(filepath, test_size, train = True, scaler=None):
    """
    Preprocesses the data to prepare for neural network.
    Returns:
        X_train_t, X_test_t, y_train_t, y_test_t, scaler
    """

    df = pd.read_csv(filepath)
    
    # removes ID, not useful for prediction
    df_mod = df.drop('ID', axis=1)

    # map ordinal / binary categoricals to ints
    df_mod["cutTemp"]        = df_mod["cutTemp"].map({"low": 0, "med": 1, "high": 2})
    df_mod["rollTemp"]       = df_mod["rollTemp"].map({"low": 0, "med": 1, "high": 2})
    df_mod["machineRestart"] = df_mod["machineRestart"].map({"no": 0, "yes": 1})

    # one-hot encode selected categoricals

    df_mod = pd.get_dummies(
        df_mod,
        columns=["alloy", "topEdgeMicroChipping", "blockSource"]
    )


    # convert all remaining bool columns to int (0/1)
    bool_cols = df_mod.select_dtypes(include="bool").columns
    df_mod[bool_cols] = df_mod[bool_cols].astype("int8")

    # split X / y
    
    if train:
        X = df_mod.drop("y_passXtremeDurability", axis=1)
        y = df_mod["y_passXtremeDurability"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    else:
        X = df_mod

    numeric_cols = ["firstPassRollPressure", "secondPassRollPressure", "clearPassNdx"]

    # fit scaler on TRAIN only, then transform both
    if scaler is None:
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])
        
        X_train = X_train.to_numpy().astype(np.float32)
        X_test  = X_test.to_numpy().astype(np.float32)
        X_train = torch.from_numpy(X_train)
        X_test  = torch.from_numpy(X_test)
    else:
        X[numeric_cols] = scaler.transform(X[numeric_cols])
        X = X.to_numpy().astype(np.float32)
        X = torch.from_numpy(X)

    if train:
        y_train = y_train.to_numpy().astype(np.float32)  # or int64 for classification
        y_test  = y_test.to_numpy().astype(np.float32)

        y_train = torch.from_numpy(y_train)
        y_test  = torch.from_numpy(y_test)

    if train:
        return X_train, X_test, y_train, y_test, scaler
    else:
        return X

def preprocess_embedding(
    filepath_train,
    filepath_val = None,
    synthetic_path=None,
    train=True,
    scaler=None,
):
    """
    When train=True:
        - If synthetic_path is given, synthetic rows are added to TRAIN.
        - VAL remains as the rows from filepath_val.

        Returns:
            X_other_train, X_other_val,
            X_embed_train, X_embed_val,
            y_train, y_val,
            scaler
    """

    df_train = pd.read_csv(filepath_train)

    if train:
        df_val = pd.read_csv(filepath_val)

        if synthetic_path is not None:
            df_synth = pd.read_csv(synthetic_path)
            df_synth = df_synth.sample(40000)
            # Order: train, synthetic, val
            df_all = pd.concat([df_train, df_synth, df_val], ignore_index=True)
            n_train_total = len(df_train) + len(df_synth)
        else:
            # Order: train, val
            df_all = pd.concat([df_train, df_val], ignore_index=True)
            n_train_total = len(df_train)
    else:
        df_all = df_train  # generic case, no val logic

    # ---------- Drop useless column ----------
    df_all_mod = df_all.drop("ID", axis=1)

    # ---------- Map ordinal / binary categoricals ----------
    df_all_mod["cutTemp"] = df_all_mod["cutTemp"].map({"low": 0, "med": 1, "high": 2})
    df_all_mod["rollTemp"] = df_all_mod["rollTemp"].map({"low": 0, "med": 1, "high": 2})
    df_all_mod["machineRestart"] = df_all_mod["machineRestart"].map({"no": 0, "yes": 1})
    df_all_mod["topEdgeMicroChipping"] = df_all_mod["topEdgeMicroChipping"].map(
        {"no": 0, "uncertain": 0.5, "yes": 1}
    )

    # ---------- Embedding categoricals ----------
    alloy_vals = sorted(df_all_mod["alloy"].unique())
    block_vals = sorted(df_all_mod["blockSource"].unique())

    alloy_map = {v: i for i, v in enumerate(alloy_vals)}
    block_map = {v: i for i, v in enumerate(block_vals)}

    df_all_mod["alloy"] = df_all_mod["alloy"].map(alloy_map)
    df_all_mod["blockSource"] = df_all_mod["blockSource"].map(block_map)

    embed_cols = ["alloy", "blockSource"]
    numeric_cols = ["firstPassRollPressure", "secondPassRollPressure", "clearPassNdx"]
    other_cols = [
        c for c in df_all_mod.columns
        if c not in embed_cols + ["y_passXtremeDurability"]
    ]

    if train:
        # ---------- Split back into train(+synthetic) / val ----------
        df_all_mod_train = df_all_mod.iloc[:n_train_total].copy()
        df_all_mod_val   = df_all_mod.iloc[n_train_total:].copy()

        X_train = df_all_mod_train.drop("y_passXtremeDurability", axis=1)
        y_train = df_all_mod_train["y_passXtremeDurability"]

        X_val   = df_all_mod_val.drop("y_passXtremeDurability", axis=1)
        y_val   = df_all_mod_val["y_passXtremeDurability"]

        # ---------- Scale numerics (fit on train only) ----------
        if scaler is None:
            scaler = StandardScaler()
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_val[numeric_cols]   = scaler.transform(X_val[numeric_cols])
        else:
            X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
            X_val[numeric_cols]   = scaler.transform(X_val[numeric_cols])

        # ---------- Split into embed + other ----------
        X_embed_train = torch.from_numpy(
            X_train[embed_cols].to_numpy().astype(np.int64)
        )
        X_embed_val = torch.from_numpy(
            X_val[embed_cols].to_numpy().astype(np.int64)
        )

        X_other_train = torch.from_numpy(
            X_train[other_cols].to_numpy().astype(np.float32)
        )
        X_other_val = torch.from_numpy(
            X_val[other_cols].to_numpy().astype(np.float32)
        )

        y_train_t = torch.from_numpy(y_train.to_numpy().astype(np.float32))
        y_val_t   = torch.from_numpy(y_val.to_numpy().astype(np.float32))

        return (
            X_other_train, X_other_val,
            X_embed_train, X_embed_val,
            y_train_t, y_val_t,
            scaler,
        )

    else:
        X = df_all_mod
        X[numeric_cols] = scaler.transform(X[numeric_cols])

        X_embed = torch.from_numpy(
            X[embed_cols].to_numpy().astype(np.int64)
        )
        X_other = torch.from_numpy(
            X[other_cols].to_numpy().astype(np.float32)
        )

        return X_other, X_embed
