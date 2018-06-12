from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np

# get the test data
raw_data = pd.read_csv('final.csv', header=None)

# split off the features
x_train = raw_data.iloc[:,:9]
# split of the classes
y_train = raw_data.iloc[:,9:10]


# create the pipeline for cross validation
pipeline = make_pipeline(MLPRegressor())

# setup the hyper params for the cross validation
hyperparameters = { 'mlpregressor__hidden_layer_sizes': [(1,), (10,), (20,), (40,), (80,)],
                    'mlpregressor__activation': ['identity', 'logistic', 'tanh', 'relu'],
                    'mlpregressor__solver': ['lbfgs', 'sgd', 'adam'],
                    'mlpregressor__learning_rate': ['constant', 'invscaling', 'adaptive']}

# cross validate using the pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

clf.fit(x_train, y_train)

score = clf.score(x_train, y_train)