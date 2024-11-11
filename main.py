print("hello World")

print("1")

import pandas as pd
from codigos.EDA import EDAProcessor

# Load the dataset
data = pd.read_csv('data/UCI_Credit_Card.csv')
print(data.head())
