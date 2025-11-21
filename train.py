import torch 
import pandas as pd

from preprocessing import preprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AluminumNeuralNetwork(nn.Module):
    def __init__(self, num_numeric_features, num_categories_list, emb_dim=4):
        """
        num_numeric_features: int, number of numeric cols
        num_categories_list: list[int], e.g. [n_alloy, n_process, ...]
        emb_dim: embedding size per categorical column (you can vary by col if you want)
        """
        super().__init__()

        # One embedding per categorical feature
        self.emb_layers = nn.ModuleList([
            nn.Embedding(num_cat, emb_dim)
            for num_cat in num_categories_list
        ])

        total_emb_dim = emb_dim * len(num_categories_list)
        input_dim = num_numeric_features + total_emb_dim

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)  # regression; use num_classes for classification

    def forward(self, x_num, x_cats):
        """
        x_num:  (B, D_num), float
        x_cats: (B, C), long, each column is a categorical ID
        """
        emb_list = []
        # For each categorical column, get its embedding
        for i, emb in enumerate(self.emb_layers):
            cat_i = x_cats[:, i]            # (B,)
            emb_i = emb(cat_i)             # (B, emb_dim)
            emb_list.append(emb_i)

        # Concatenate all embeddings
        if emb_list:
            x_cat = torch.cat(emb_list, dim=1)   # (B, total_emb_dim)
            x = torch.cat([x_num, x_cat], dim=1)
        else:
            x = x_num

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


if __name__ == "__main__":
    filepath = 'aluminum_coldRoll_train.csv'
    preprocess(filepath)
    # 1. Define the loss function
    criterion = nn.BCELoss()

    # 2. Define the optimizer
    optimizer = optim.Adam(model_pytorch.parameters(), lr=0.01)

    # 3. Set the number of training epochs
    epochs = 10