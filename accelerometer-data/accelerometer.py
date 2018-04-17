import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('1.csv', delimiter=',', names=['id','x','y','z','label'])

# get each column

raw_x = np.array(data['x'])
raw_y = np.array(data['y'])
raw_z = np.array(data['z'])

# normalize the data for each column

norm_x = raw_x / np.linalg.norm(raw_x)
norm_y = raw_y / np.linalg.norm(raw_y)
norm_z = raw_z / np.linalg.norm(raw_z)

plt.subplot(3,1,1)
plt.plot(norm_x)

plt.subplot(3,1,2)
plt.plot(norm_y)

plt.subplot(3,1,3)
plt.plot(norm_z)

plt.tight_layout()
plt.show()