import torch.nn as nn
from torch.utils.data import Dataset
import torch

# Define the PyTorch neural network model
class BinaryClassifier(nn.Module):
    def __init__(self, input):
        super(BinaryClassifier, self).__init__()
        self.layer_1 = nn.Linear(input.shape[1], 32)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(32, 64)
        self.layer_3 = nn.Linear(64, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_5 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        x = self.relu(self.layer_5(x))
        x = self.sigmoid(self.layer_out(x))
        return x
    

class TabularDataset(Dataset):
    def __init__(self, X_num, X_cats, y = None):
        self.X_num = X_num       # (N, D_num)
        self.X_cats = X_cats     # (N, C)
        self.y = y        # (N,)

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_num[idx], self.X_cats[idx], self.y[idx]
        else:
            return self.X_num[idx], self.X_cats[idx]

# early stopping
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            if score < self.best_score:
                self.best_model_state = model.state_dict()
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
    
# Define the PyTorch neural network model
class EmbeddingClassifier(nn.Module):
    def __init__(self, input):
        super(EmbeddingClassifier, self).__init__()

        self.alloy_embedding = nn.Embedding(11, 4)
        self.block_embedding = nn.Embedding(4, 2)

        self.layer_1 = nn.Linear(input.shape[1]+4+2, 32)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(32, 64)
        self.layer_3 = nn.Linear(64, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_5 = nn.Linear(64, 32)
        #self.layer_6 = nn.Linear(128, 64)
        #self.layer_7 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_embed):
        """
        x:        (batch, D_other) float tensor
        x_embed:  (batch, 2) long/int tensor -> [alloy_id, blockSource_id]
        """

        # Make sure x_embed is long for nn.Embedding
        x_embed = x_embed.long()

        # Get per-example embeddings
        x_alloy = self.alloy_embedding(x_embed[:, 0])   # (batch, emb_dim_alloy)
        x_block = self.block_embedding(x_embed[:, 1])   # (batch, emb_dim_block)

        # Concatenate along feature dimension
        x = torch.cat([x, x_alloy, x_block], dim=1)     # (batch, D_other + emb_alloy + emb_block)

        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        x = self.relu(self.layer_5(x))
        #x = self.relu(self.layer_6(x))
       # x = self.relu(self.layer_7(x))
        x = self.sigmoid(self.layer_out(x))
        return x