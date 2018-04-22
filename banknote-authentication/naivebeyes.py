import numpy as np
from sklearn.naive_bayes import GaussianNB

train_data = np.genfromtxt("banknote_train.csv", delimiter=",")

test_data = np.genfromtxt("banknote_test.csv", delimiter=",")

X = train_data[:,:-1]
y = train_data[:,-1]

gnb = GaussianNB()

y_pred = gnb.fit(X, y).predict(X)

test = gnb.predict(test_data)

print test

