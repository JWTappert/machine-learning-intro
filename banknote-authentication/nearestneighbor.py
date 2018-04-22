import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# read in training data
train_data = np.genfromtxt("banknote_train.csv", delimiter=",")

# read in test data
test_data = np.genfromtxt("banknote_test.csv", delimiter=",")

# split data points from classes in training data
X = train_data[:,:-1]
# get only the classes that match the data
y = train_data[:,-1]

# train the knn model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)

# predict the test data
print(knn.predict(test_data))
print(knn.predict_proba(test_data))
