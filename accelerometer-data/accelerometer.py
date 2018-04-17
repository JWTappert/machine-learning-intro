import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('1.csv', delimiter=',', names=['id','x','y','z','label'])

print data

# get each column

raw_x = np.array(data['x'])
raw_y = np.array(data['y'])
raw_z = np.array(data['z'])

# normalize the data for each column

norm_x = 