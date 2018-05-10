import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
# get the qualities
qualities = train_data['quality']
# get the classes and convert them to digits, i.e. 0 = red 1 = white
types = pd.factorize(train_data['class'])[0]

# classify

# create our random forest classifier
clf1 = RandomForestClassifier()
# classify the type
clf1.fit(train_data[features], types)
type_pred = clf1.predict(test_data)
# create new classifier
clf2 = RandomForestClassifier()
# classify the quality
clf2.fit(train_data[features], qualities)
qual_pred = clf2.predict(test_data)

results = pd.DataFrame(columns=['id','quality','type'])
results['quality'] = qual_pred.flatten()
results['type'] = type_pred.tolist()

results.to_csv('tappert.csv', sep=' ')