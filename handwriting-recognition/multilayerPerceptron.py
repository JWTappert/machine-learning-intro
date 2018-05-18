from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np

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

clf = MLPClassifier()
clf.fit(x_train, y_train)

# get the prediction
pred = clf.predict(test_data)

# create a new dataframe for our results
results = pd.DataFrame(columns=['id', 'prediction'])
            
rows = pred.shape[0]
cols = pred.shape[1]
pred_array = np.empty((319,1))

for i in range(0, rows):
    for j in range(0, cols):
        if pred[i,j] == 1:
            pred_array[i] = j

results['prediction'] = pred_array.flatten()
results.to_csv('tappert.csv', sep=' ')