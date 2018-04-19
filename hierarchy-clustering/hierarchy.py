import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

data = np.random.randint(100, size=10)

Z = hierarchy.linkage(data, 'ward')
plt.figure()
dn = hierarchy.dendrogram(Z)
plt.savefig('hierarchy.png')