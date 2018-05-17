from sklearn.neural_network import MLPClassifier
import pandas as pd

# get the test data
raw_test = pd.read_csv('semeion-test.data', delim_whitespace=True, header=None)
# get IDs from test_data
ids = raw_test.iloc[:, :1]
# get the actual training data
test_data = raw_test.iloc[:, 1:]

# get the training data
raw_data = pd.read_csv('semeion-train.data', delim_whitespace=True, header=None)
# split off the features
x_train = raw_data.iloc[:, :256]
# split of the classes
y_train = raw_data.iloc[:, 256:]

# create our classification object
clf = MLPClassifier()

clf.fit(x_train, y_train)

pred = clf.predict(test_data)
