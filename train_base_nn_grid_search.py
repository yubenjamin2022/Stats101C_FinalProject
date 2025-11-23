import torch
import torch.nn as nn
from preprocessing import preprocess
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
import tqdm
import pandas as pd
from itertools import product

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

if __name__ == "__main__":

    # ---------------------------
    # Hyperparameter grids (edit these)
    # ---------------------------
    test_sizes    = [0.1]
    num_epochs_ls = [30]
    batch_sizes   = [32, 64, 128]
    learning_rates = [1e-2, 1e-3, 1e-4]
    gammas        = [0.9, 0.95, 0.99]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "aluminum_coldRoll_train.csv"

    results = []

    # Cartesian product of all hyperparameter combinations
    for test_size, num_epochs, batch_size, LR, gamma in product(
        test_sizes, num_epochs_ls, batch_sizes, learning_rates, gammas
    ):
        print(
            f"\n=== New config === "
            f"test_size={test_size}, num_epochs={num_epochs}, "
            f"batch_size={batch_size}, LR={LR}, gamma={gamma}"
        )

        # ---------------------------
        # Data prep for this config
        # ---------------------------
        X_train, X_test, y_train, y_test, scaler = preprocess(
            data_path, test_size
        )

        # Instantiate model fresh for each combo
        model = BinaryClassifier(X_train).to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = ExponentialLR(optimizer, gamma=gamma)

        train_ds = TensorDataset(X_train, y_train)
        test_ds  = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # ---------------------------
        # Training loop
        # ---------------------------
        best_val_loss = float("inf")
        best_epoch = -1

        for epoch in range(num_epochs):
            # ---- train ----
            model.train()
            running_loss = 0.0

            for X, y in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                prediction = model(X)               # shape (B, 1)
                loss = criterion(prediction, y.unsqueeze(1))  # y: (B,) -> (B,1)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)

            # ---- eval ----
            model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for X, y in test_loader:
                    X = X.to(device)
                    y = y.to(device)
                    prediction = model(X)
                    loss = criterion(prediction, y.unsqueeze(1))
                    eval_loss += loss.item()

            avg_eval_loss = eval_loss / len(test_loader)

            # track best validation loss
            if avg_eval_loss < best_val_loss:
                best_val_loss = avg_eval_loss
                best_epoch = epoch + 1  # 1-based indexing

            scheduler.step()

            print(
                f"[Config ts={test_size}, ne={num_epochs}, bs={batch_size}, "
                f"lr={LR}, g={gamma}] "
                f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                f"val_loss={avg_eval_loss:.4f}"
            )

        # ---------------------------
        # Save result for this combo
        # ---------------------------
        results.append({
            "test_size": test_size,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": LR,
            "gamma": gamma,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        })

    # ---------------------------
    # Save all results to CSV
    # ---------------------------
    df_results = pd.DataFrame(results)
    df_results.to_csv("grid_search_results.csv", index=False)
    print("\nGrid search complete. Results saved to grid_search_results.csv")
