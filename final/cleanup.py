import numpy as np
import pandas as pd

# read in the data
data = pd.read_csv('flybook-data.csv')

# drop all rows where there is a NaN
data = data.dropna()

# drop all rows that have a 0 value for total revenue
data = data[data.TotalRev != 0]




data.to_csv('raw-data.csv')