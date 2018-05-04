import numpy as np
import pandas as pd
import scipy.stats as stats

train_data = pd.read_csv('auto-mpg-train.data', delim_whitespace=True)

# TODO: implement the linear regression workflow