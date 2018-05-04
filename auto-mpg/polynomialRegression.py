import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# get the train data
train_data = pd.read_csv('auto-mpg-train.data', delim_whitespace=True, header=None)
# replace NaN with the mean of that column
train_data.fillna(train_data.mean(), inplace=True)
# convert pandas dataframe to numpy array
train_data = train_data.values
# remove the string car name from the array
train_data = train_data[:,:-1]
# remove the mpg column
y = train_data[:,0]
y = y.reshape(-1,1)
# remove the mpg column from the train data
X = train_data[:,1:]


# get the test data
test_data = pd.read_csv('auto-mpg-test.data', delim_whitespace=True, header=None)
# replace NaN with the mean of that column
test_data.fillna(test_data.mean(), inplace=True)
# convert pandas dataframe to numpy array
test_data = test_data.values
# remove the string car names from the array
test_data = test_data[:,:-1]
# get the ids from the test data
ids = test_data[:,0]
# remove the id column from the test data
x = test_data[:,1:]


# Create our pieline that will take our data and send it to the polynomial
# features instance first then the linear regression instance.
# this will allow us to use a polynomial feature set with a linear regression
# algorithm
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)

# call predict to generate our output
pred = model.predict(x)

# throw the columns together for the output file
results = np.column_stack((ids, pred))
# save to csv
np.savetxt('tappert_poly.csv', results, delimiter=',', fmt='%f')