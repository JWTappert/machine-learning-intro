from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np

# get the test data
raw_data = pd.read_csv('final.csv', header=None)

mask = np.random.rand(len(raw_data)) < 0.8

train_data = raw_data[mask]

test_data = raw_data[~mask]

x_test = test_data.iloc[:,:9]

y_test = test_data.iloc[:,9:10]


# split off the features
x_train = train_data.iloc[:,:9]
# split of the classes
y_train = train_data.iloc[:,9:10]

clf = MLPRegressor()

clf.fit(x_train, y_train)

pred = clf.predict(x_test)

