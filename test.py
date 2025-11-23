import pandas as pd

df = pd.read_csv('aluminum_coldRoll_train.csv')
df1 = pd.read_csv('synthetic_data_10.csv')
df2 = pd.read_csv('synthetic_data_20.csv')
df3 = pd.read_csv('synthetic_data_30.csv')

discrete_columns = [
    'alloy',
    'cutTemp',
    'rollTemp',
    'topEdgeMicroChipping',
    'blockSource',
    'machineRestart',
    'contourDefNdx',
]

for column in discrete_columns:
    print(column)
    print(sorted(df[column].unique()))
    print(sorted(df1[column].unique()))
    print(sorted(df2[column].unique()))
    print(sorted(df3[column].unique()))
    print('-------------------------------')