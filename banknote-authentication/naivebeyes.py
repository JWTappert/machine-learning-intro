import numpy as np
from sklearn.naive_bayes import GaussianNB

# read in training data
train_data = np.genfromtxt("banknote_train.csv", delimiter=",")

# read in the test data
test_data = np.genfromtxt("banknote_test.csv", delimiter=",")

# seperate the training data from the classes
X = train_data[:,:-1]
# get only the classes
y = train_data[:,-1]

# train the NB model
gnb = GaussianNB()
y_pred = gnb.fit(X, y).predict(X)

# predict the test data
test = gnb.predict(test_data)

print test

