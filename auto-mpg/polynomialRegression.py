import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# get the train data
train_data = pd.read_csv('auto-mpg-train.data', delim_whitespace=True, header=None)

# replace NaN with the average of that column
train_data.fillna(train_data.mean(), inplace=True)

# convert pandas dataframe to numpy array
train_data = train_data.values
# remove the string car name from the array
train_data = train_data[:,:-1]

# Create our Polynomial Features object and fit the data
# this will allow us to use a polynomial feature set with a inear regression algorithm
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(train_data)

print X