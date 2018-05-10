import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Get the red wine data
red_wine_train = pd.read_csv('wine-quality-red-train.csv', delimiter=';')

# Get the white wine data
white_wine_train = pd.read_csv('wine-quality-white-train.csv', delimiter=';')

# add class column to red wine training set
red_wine_train['class'] = 'red'

# add class column to white wine training set
white_wine_train['class'] = 'white'

# concatenate the two dataframes together
train_data = pd.concat([red_wine_train, white_wine_train])

# k folds cross validate to help improve our model
train, validation = train_test_split(train_data, test_size=0.50, random_state=5)

# get a list of the features
features = train.columns[:12]

# convert classes into digits
train_y = pd.factorize(train['class'])[0]
valid_y = pd.factorize(validation['class'])[0]

# create our random forest classifier
clf = RandomForestClassifier()

clf.fit(train[features], train_y)

print clf.score(validation[features], valid_y)