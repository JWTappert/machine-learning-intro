import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# get the train data
train_data = pd.read_csv('auto-mpg-train.data', delim_whitespace=True, header=None)
# replace NaN with the mean of that column
train_data.fillna(train_data.mean(), inplace=True)

# get the test data
test_data = pd.read_csv('auto-mpg-test.data', delim_whitespace=True, header=None)
# replace NaN with the mean of that column
test_data.fillna(test_data.mean(), inplace=True)

# convert pandas dataframe to numpy array
train_data = train_data.values
# remove the string car name from the array
train_data = train_data[:,:-1]
# convert pandas dataframe to numpy array
test_data = test_data.values
# remove the string car names from the array
y = test_data[:,:-1]

# Create our Polynomial Features object and fit the data
# this will allow us to use a polynomial feature set with a linear regression algorithm
poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(train_data)

clf = LinearRegression()
clf.fit(X_, y)

