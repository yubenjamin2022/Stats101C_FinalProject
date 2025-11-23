import torch
import torch.nn as nn
from preprocessing import preprocess_embedding
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
import tqdm
import pandas as pd
from models import EmbeddingClassifier, TabularDataset, EarlyStopping

if __name__ == "__main__":

    # relevant hyperparameters
    test_size = 0.1
    device = 'cuda'
    num_epochs = 100
    batch_size = 128
    LR = 0.01
    gamma = 0.99

    # scheduler hyperparameters
    patience = 15
    delta = 0.01

    # file names
    train_set = './data/aluminum_coldRoll_train.csv'
    val_set = './data/aluminum_coldRoll_val.csv'
    synthetic_data = './data/synthetic_data_30.csv'
    test_set = './data/aluminum_coldRoll_testNoY.csv'
    return_file = './predictions/Group_8_predictions_synthetic_30_40000.csv'

    X_train, X_test, \
    X_embed_train, X_embed_test, \
    y_train, y_test, scaler = preprocess_embedding(train_set, val_set, synthetic_data, test_size)

    # Instantiate the model
    model = EmbeddingClassifier(X_train).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    early_stopping = EarlyStopping(patience=patience, delta=delta)

    train_ds = TabularDataset(X_train, X_embed_train, y_train)
    test_ds  = TabularDataset(X_test, X_embed_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # training loop
    for epoch in range(num_epochs):

        runningLoss = 0

        model.train()
        for X, X_embed, y in tqdm.tqdm(train_loader):
            X = X.to(device)
            X_embed = X_embed.to(device)
            y = y.to(device)
            prediction = model.forward(X, X_embed)
            currentLoss = criterion(prediction, y.unsqueeze(1))
            currentLoss.backward()

            optimizer.step()
            optimizer.zero_grad()

            runningLoss += currentLoss.item()

        model.eval()
        eval_loss = 0
        for X, X_embed, y in test_loader:
            X = X.to(device)
            X_embed = X_embed.to(device)
            y = y.to(device)
            prediction = model.forward(X, X_embed)
            currentLoss = criterion(prediction, y.unsqueeze(1))

            eval_loss += currentLoss.item()
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Loss: {runningLoss/len(train_loader)}, Eval Loss: {eval_loss/len(test_loader)}')
        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    early_stopping.load_best_model(model)

    # run on test data

    X_test_final, X_embed_final = preprocess_embedding(test_set, train = False, scaler = scaler)
    final_test = TabularDataset(X_test_final, X_embed_final)
    final_test_loader  = DataLoader(final_test, batch_size=1, shuffle=False)

    predictions = []
    with torch.no_grad():
        for X, X_embed in tqdm.tqdm(final_test_loader):
            X = X.to(device)
            X_embed = X_embed.to(device)
            prediction = model.forward(X, X_embed)
            predictions.append(prediction.item())

    df_ex = pd.read_csv(test_set)
    df_final = pd.DataFrame({'ID': df_ex['ID'], 'y_passXtremeDurability': predictions})
    df_final.to_csv(return_file, index = False)


