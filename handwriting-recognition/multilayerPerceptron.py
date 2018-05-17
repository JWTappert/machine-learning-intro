from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
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

pipeline = make_pipeline(MLPClassifier())

# cross validate using the pipeline
clf = GridSearchCV(pipeline, cv=10)
# train the model with subset of data
clf.fit(x_train, y_train)

# get the prediction
pred = clf.predict(test_data)
