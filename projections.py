import math
from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import seaborn as sb
import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator

def get_n_components(x):
    n_comps = math.floor(len(x.iloc[0]) / 2)
    print ("Number of components: " + str(n_comps))

    return n_comps

# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
def get_num_pca_comp(x, name):
    pca = PCA().fit(x)

    vr = np.cumsum(pca.explained_variance_ratio_)

    x = range(1, len(vr) + 1)
    kneedle = KneeLocator(x, vr, S=1.0, curve='concave', direction='increasing')
    
    knee = math.ceil(kneedle.knee)

    plt.figure()
    plt.plot(x, vr, 'bx-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.axvline(x=knee, label="Selected Number of Components")
    plt.legend(loc="best")
    plt.title('N-Components vs. Explained Variance for {}'.format(name))
    plt.savefig("pca_curves/{}.png".format(name))

    return knee

def run_pca(x, _y, name):
    n_components = get_num_pca_comp(x, name)
    pca = PCA(n_components=n_components)

    transformer = pca.fit(x)

    return transformer.transform(x)

def run_ica(x, _y, name):
    n_components = get_n_components(x) + 1
    ica = FastICA(n_components=n_components, whiten=True)

    transformer = ica.fit(x)

    return transformer.fit_transform(x)

def run_rp(x, _y, name):
    n_components = get_n_components(x)
    transformer = random_projection.GaussianRandomProjection(n_components=n_components)

    return transformer.fit_transform(x)

# https://scikit-learn.org/stable/modules/feature_selection.html
def run_tree_selection(x, y, name):
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(x, y)

    transformer = SelectFromModel(clf, prefit=True)
    
    return transformer.transform(x)