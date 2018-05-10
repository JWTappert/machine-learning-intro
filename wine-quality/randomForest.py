import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


# Get the red wine data
red_wine_train = pd.read_csv('wine-quality-red-train.csv', delimiter=';')

# Get the white wine data
white_wine_train = pd.read_csv('wine-quality-white-train.csv', delimiter=';')

# get the test data
test_data = pd.read_csv('wine-quality-test.csv', delimiter=';')

# add class column to red wine training set
red_wine_train['class'] = 'red'

# add class column to white wine training set
white_wine_train['class'] = 'white'

# concatenate the two dataframes together
train_data = pd.concat([red_wine_train, white_wine_train])

# get the features of the data for indexing
features = train_data.columns[:12]

# get the classes and convert them to digits, i.e. 0 = red 1 = white
y = pd.factorize(train_data['class'])[0]

# k folds cross validate to help improve our model
#kf = KFold(n_splits=5)
#kf.get_n_splits(train_data)

# create our random forest classifier
clf = RandomForestClassifier()

#for train_index, test_index in kf.split(train_data):
#    x_train, x_test = train_data[train_index], train_data[test_index]
#    y_train, y_test = y[train_index], y[test_index]
#    clf.fit(x_train, y_train)
#    print clf.predict(x_test, y_test)

clf.fit(train_data[features], y)
print clf.predict(test_data)
