import matplotlib.pyplot as plt
import sklearn

# load the iris dataset from the scikit learn module
iris_dataset = sklearn.datasets.load_iris()

# get septal and petal lengths and widths
x = iris_dataset.data

# get the classes as numbers i.e. 'setosa': 0 , 'versicolor': 1 , 'virginica': 2
y = iris_dataset.target

# get class name strings
names = iris_dataset.target_names

# use the pca module to reduce the demensionality of the iris data
pca = sklearn.decomposition.PCA()
pca.fit(x)
pca_x = pca.transform(x)

# create tuple of the color, class, and its corresponding num value
colors = ['red', 'green', 'blue']
color_to_class = zip(colors, [0, 1, 2], names)

plt.figure(figsize=(8,6))
# map the tuple to the data in the scatter plot
for color, i, name in color_to_class:
    # plot first two primary components for each iris class
    plt.scatter(pca_x[y == i, 0], pca_x[y == i, 1], color=color, label=name)
    
# prettify the graph
plt.legend(shadow=False)
plt.title('Principal Component Analysis on Iris dataset')
plt.show()