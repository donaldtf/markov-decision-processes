from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import math
import numpy as np
from kneed import DataGenerator, KneeLocator

# https://pythonprogramminglanguage.com/kmeans-elbow-method/
def get_optimal_k(name, X):
    # k means determine k
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        # distortions.append(kmeanModel.inertia_)

    distortions = distortions / distortions[0]

    kneedle = KneeLocator(K, distortions, S=1.0, curve='convex', direction='decreasing')
    knee = math.floor(kneedle.knee)

    # Plot the elbow
    plt.figure()
    plt.plot(K, distortions, 'bx-')
    plt.axvline(x=knee, label="Selected K")
    plt.legend(loc="best")
    plt.xlabel('k')
    plt.ylabel('Normalized Average Distance')
    plt.title('Elbow Curve for {}'.format(name))
    plt.savefig("elbow_curves/{}.png".format(name))

    return knee

def kmeans(name, x):
    optimal_k = get_optimal_k(name, x)
    kmeans = KMeans(n_clusters=optimal_k, random_state=99).fit(x)

    labels = list(kmeans.labels_)

    print ("cluster_centers_")
    print (kmeans.cluster_centers_)
    print ("selected k: " + str(optimal_k))

    return np.reshape(kmeans.labels_, (-1, 1))

# Credit to: https://github.com/cmaron/CS-7641-assignments/blob/master/assignment3/experiments/clustering.py
def gmm(name, x):
    gmm = GaussianMixture(n_components=4, random_state=99)

    gmm.fit(x)
    gmm_labels = gmm.predict(x)

    print ("GMM Labels")
    print (gmm_labels)

    return np.reshape(gmm_labels, (-1, 1))
