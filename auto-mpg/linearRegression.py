import numpy as np
import pandas as pd
import scipy.stats as stats

train_data = pd.read_csv('auto-mpg-train.data', delim_whitespace=True)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(train_data)

# test_data = np.genfromtxt('auto-mpg-test.data', delimiter=" ")
