import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Get the red wine data
red_wine_train = pd.read_csv('wine-quality-red-train.csv', delimiter=';')
# Get the white wine data
white_wine_train = pd.read_csv('wine-quality-white-train.csv', delimiter=';')

# add class column to red wine training set
red_with_class = red_wine_train.copy()
red_with_class['class'] = 'red'

# add class column to white wine training set
white_with_class = white_wine_train.copy()
white_with_class['class'] = 'white'

# concatenate our data sets together
q_train_data = pd.concat([red_wine_train, white_wine_train])
t_train_data = pd.concat([red_with_class, white_with_class])

# get the test data
test_data = pd.read_csv('wine-quality-test.csv', delimiter=';')
# Get the IDs only
IDs = test_data['ID']
# then remove them from the dataset
test_data = test_data.drop('ID', axis=1)

# get types and feature data seperated
type_y = pd.factorize(t_train_data['class'])[0]
type_x = t_train_data.drop('class', axis=1)

# get qualities and features seperated
qual_y = q_train_data['quality']
qual_x = q_train_data.drop('quality', axis=1)

# create the pipeline object to feed to the cross validator
pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))

# fine tune the paramters for the CV
hyperparameters = { 'randomforestclassifier__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestclassifier__max_depth': [None, 5, 3, 1]}

# cross validate using the pipeline
quality_clf = GridSearchCV(pipeline, hyperparameters, cv=10)
# train the model with subset of data
quality_clf.fit(qual_x, qual_y)
#predict the quality
quality_pred = quality_clf.predict(test_data)
# append the quality prediction to the test data so the columns match
test_data['quality'] = quality_pred

# create new classifier for the type prediction
type_clf = GridSearchCV(pipeline, hyperparameters, cv=10)
# fit the data with the quality appended
type_clf.fit(type_x, type_y)
# predict the type
type_pred = type_clf.predict(test_data)

results = pd.DataFrame(columns=['id','quality','type'])
results['quality'] = quality_pred.flatten()
results['type'] = type_pred.tolist()

results.to_csv('tappert.csv', sep=' ')