import numpy as np
import pandas as pd

data = pd.read_csv('flybook-data.csv')

data2 = data.dropna()

data2.to_csv('raw-data.csv')