import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Get the red wine data
red_wine_train = pd.read_csv('wine-quality-red-train.csv', delimiter=';')
# Get the white wine data
white_wine_train = pd.read_csv('wine-quality-white-train.csv', delimiter=';')
# get the test data
test_data = pd.read_csv('wine-quality-test.csv', delimiter=';')

# process the data

# add class column to red wine training set
red_wine_train['class'] = 'red'
# add class column to white wine training set
white_wine_train['class'] = 'white'
# concatenate the two dataframes together
train_data = pd.concat([red_wine_train, white_wine_train])
# get the features of the data for indexing
features = train_data.columns[:12]

# get types and feature data seperated
type_y = pd.factorize(train_data['class'])[0]
type_x = train_data[features]

# split the data into train and test sets to see how well our model does before actually predicting
X_train, X_test, y_train, y_test = train_test_split(type_x, type_y, test_size=0.2, random_state=123, stratify=type_y)

# create the pipeline object to feed to the cross validator
pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))

# fine tune the paramters for the CV
hyperparameters = { 'randomforestclassifier__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestclassifier__max_depth': [None, 5, 3, 1]}

# cross validate using the pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

# train the model with subset of data
clf.fit(X_train, y_train)

# predict on the test data
pred = clf.predict(X_test)
# print out scores
print r2_score(y_test, pred)
# out: 0.9792664431673053
print mean_squared_error(y_test, pred)
# out: 0.0038498556304138597

