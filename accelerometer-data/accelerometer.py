import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

data = pd.read_csv('data/1.csv', delimiter=',', names=['id','x','y','z','label'])

# simple function to transform the hz to seconds for the x axis labels
def hzToSeconds(val, pos, hz=52):
    return int(val / hz)

# set axis formatter to our function
formatter = FuncFormatter(hzToSeconds)

# group the data by activity label
grpd_data = data.groupby('label')

# create index to correctly subplot the activities
index = 1
# dictionary to map index to activity
act_to_index = {1:"Working at a Computer", 2:"Standing up, Walking and Going up/down Stairs", 3:"Standing", 4:"Walking", 5:"Going up/down Stairs", 6:"Walking and Talking with Someone", 7:"Talking While Standing"}

plt.figure(figsize=(20,40))
for key, item in grpd_data:
    # get x, y, and z into numpy array and then normalize the data
    xarray = np.array(item['x'])
    x = xarray / np.linalg.norm(xarray)
    # shift x up in order to seperate the lines
    x = x + np.mean(x)
    yarray = np.array(item['y'])
    y = yarray / np.linalg.norm(yarray)
    zarray = np.array(item['z'])
    z = zarray / np.linalg.norm(zarray)
    # shift z down to seperate the lines
    z = z - np.mean(z)
    
    # plot each activity as a subplot
    plt.subplot(7,1,index)
    plt.plot(x, color="red")
    plt.plot(y, color="green")
    plt.plot(z, color="blue")
    plt.xlabel("time (s)")
    plt.ylabel("normalized magnitude")
    plt.title(act_to_index[index])
    ax = plt.gca()
    ax.xaxis.set_major_formatter(formatter)
    index += 1
    

plt.tight_layout()
plt.show()