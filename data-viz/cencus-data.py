import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('adult.data', 
                       delimiter=',', 
                       names=[
                               'age',
                               'workclass',
                               'fnlwgt',
                               'education',
                               'education-num',
                               'marital-status',
                               'occupation',
                               'relationship',
                               'race',
                               'sex',
                               'capital-gain',
                               'capital-loss',
                               'hours-per-week',
                               'native-country',
                               'salary'
                               ])
print raw_data
pd.set_option('display.height', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 15)

age = np.array(raw_data['age'])
edu = np.array(raw_data['education-num'])

plt.plot(age, edu)
plt.show()
