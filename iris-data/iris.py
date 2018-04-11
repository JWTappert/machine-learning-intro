import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA

# load the iris dataset from the scikit learn module
iris_dataset = datasets.load_iris()

# get septal and petal lengths and widths
x = iris_dataset.data

# get the classes as numbers i.e. 'setosa': 0 , 'versicolor': 1 , 'virginica': 2
y = iris_dataset.target

# get class name strings
names = iris_dataset.target_names

pca = PCA(n_components=2)
pca.fit(x)
pca_x = pca.transform(x)

plt.figure()
colors = ['red', 'green', 'blue']
lw = 2

for color, i, name in zip(colors, [0, 1, 2], names):
    plt.scatter(pca_x[y == i, 0], pca_x[y == i, 1], color=color, alpha=.8, lw=lw,
                label=name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.show()