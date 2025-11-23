import torch
import torch.nn as nn
from preprocessing import preprocess, TabularDataset
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
import tqdm
import pandas as pd
from models import BinaryClassifier

if __name__ == "__main__":

    # relevant hyperparameters
    test_size = 0.1
    device = 'cuda'
    num_epochs = 25
    batch_size = 128
    LR = 0.01
    gamma = 0.99

    X_train, X_test, y_train, y_test, scaler = preprocess('aluminum_coldRoll_train.csv', test_size)

    # Instantiate the model
    model = BinaryClassifier(X_train).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # training loop
    for epoch in range(num_epochs):

        runningLoss = 0

        model.train()
        for X, y in tqdm.tqdm(train_loader):
            X = X.to(device)
            y = y.to(device)
            prediction = model.forward(X)
            currentLoss = criterion(prediction, y.unsqueeze(1))
            currentLoss.backward()

            optimizer.step()
            optimizer.zero_grad()

            runningLoss += currentLoss.item()

        model.eval()
        eval_loss = 0
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            prediction = model.forward(X)
            currentLoss = criterion(prediction, y.unsqueeze(1))

            eval_loss += currentLoss.item()
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Loss: {runningLoss/len(train_loader)}, Eval Loss: {eval_loss/len(test_loader)}')

    # run on test data

    X_test_final= preprocess('aluminum_coldRoll_testNoY.csv', 0, False, scaler)
    final_test = TensorDataset(X_test_final)
    final_test_loader  = DataLoader(final_test, batch_size=1, shuffle=False)

    predictions = []
    with torch.no_grad():
        for (X,) in final_test_loader:
            X = X.to(device)
            prediction = model.forward(X)
            predictions.append(prediction.item())

    df_ex = pd.read_csv('aluminum_coldRoll_testNoY.csv')
    df_final = pd.DataFrame({'ID': df_ex['ID'], 'y_passXtremeDurability': predictions})
    df_final.to_csv('Group_8_predictions.csv', index = False)


