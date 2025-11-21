import pandas as pd

import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, X_num, X_cats, y):
        self.X_num = torch.from_numpy(X_num).float()       # (N, D_num)
        self.X_cats = torch.from_numpy(X_cats).long()      # (N, C)
        self.y = torch.from_numpy(y).float()               # (N,)

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cats[idx], self.y[idx]

def preprocess(filepath):

    """
    
    Preprocesses the data to prepare for neural network.
    
    """

    df = pd.read_csv(filepath)
    
    # removes ID, not useful for prediction
    df_mod = df.drop('ID', axis = 1)

    # adjusts (some) categorical variables
    df_mod['cutTemp'] = df_mod['cutTemp'].replace({'low': 0, 'med': 1, 'high': 2})
    df_mod['rollTemp'] = df_mod['rollTemp'].replace({'low': 0, 'med': 1, 'high': 2})
    df_mod['machineRestart'] = df_mod['machineRestart'].replace({'no': 0, 'yes': 1})

    # TO DO: create embedding layers for some of the 

    # sample code to try one hot encodings
    one_hot_encoded_df = pd.get_dummies(df, columns=['cutTemp', 'alloy'])
    print(one_hot_encoded_df)

    X = df_encoded.drop('y_passXtremeDurability', axis=1)
    y = df_encoded['y_passXtremeDurability']