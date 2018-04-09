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

# get average education level by race
races = np.array(raw_data['race'])

# get the indexes of each race
white_index = races == ' White'
black_index = races == ' Black'
indian_index = races == ' Amer-Indian-Eskimo'
asian_index = races == ' Asian-Pac-Islander'
other_index = races == ' Other'

# get the count of participants by race
white_count = Counter(races[white_index])
black_count = Counter(races[black_index])
indian_count = Counter(races[indian_index])
asian_count = Counter(races[asian_index])
other_count = Counter(races[other_index])

# plot a bar chart of the participants
plt.bar(["Other", "Amer-Indian", "Asian", "Black", "White"], [other_count[' Other'], indian_count[' Amer-Indian-Eskimo'], asian_count[' Asian-Pac-Islander'], black_count[' Black'], white_count[' White']])
plt.ylabel("Count")
plt.xlabel("Participant Race")
plt.title("Survey Participants by Race")
plt.show()
