import numpy as np
from sklearn.neighbors import KNeighborsClassifier

train_data = np.genfromtxt("banknote_train.csv", delimiter=",")

test_data = np.genfromtxt("banknote_test.csv", delimiter=",")

X = train_data[:,:-1]
y = train_data[:,-1]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)

print(knn.predict(test_data))
print(knn.predict_proba(test_data))
