from ctgan import CTGAN
from ctgan import load_demo
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

df = pd.read_csv('./data/aluminum_coldRoll_train.csv')
df = df.drop('ID', axis=1)

# Names of the columns that are discrete
discrete_columns = [
    'alloy',
    'cutTemp',
    'rollTemp',
    'topEdgeMicroChipping',
    'blockSource',
    'machineRestart',
    'contourDefNdx',
]

for epoch in [10, 20, 30]:
    ctgan = CTGAN(epochs=epoch)
    ctgan.fit(df, discrete_columns)

    # Create synthetic data
    synthetic_data = ctgan.sample(150000)
    synthetic_data['ID'] = list(range(150000))

    synthetic_data.to_csv(f'./data/synthetic_data_{epoch}.csv', index = False)