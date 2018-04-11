import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# setup pandas to pretty print data
pd.set_option('display.height', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 15)

# read in data from adult.data and give names to each column for easier column access
raw_data = pd.read_csv('iris.data', delimiter=',', names=['sepal-length','sepal-width','petal-length','petal-width','class'])

print raw_data