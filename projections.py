import math
from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sb
import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from scipy.stats import kurtosis
from sklearn.metrics import f1_score

def get_n_components(x):
    n_comps = math.floor(len(x.iloc[0]) / 2)
    print ("Number of components: " + str(n_comps))

    return n_comps

# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
def get_num_pca_comp(x, name):
    pca = PCA().fit(x)

    vr = np.cumsum(pca.explained_variance_ratio_)
    print ("Distribution of Eigen Values")
    print(pca.explained_variance_)

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

def get_num_ica_comp(x, y, name):
    pca = PCA().fit(x)

    scores = []
    K = range(1,7)
    for n_components in K:
        ica = FastICA(n_components=n_components)

        transformer = ica.fit(x)

        new_x = transformer.fit_transform(x)

        clf = GradientBoostingClassifier(learning_rate=0.25, max_depth=5, n_estimators=30)
        clf.fit(new_x, y)

        y_pred = clf.predict(new_x)

        f1 = f1_score(y, y_pred)

        scores.append(f1)

    n_comps = np.argmax(scores) + 1

    print ("ICA")
    print (scores)
    print (n_comps)

    return n_comps

def run_pca(x, _y, name):
    n_components = get_num_pca_comp(x, name)
    pca = PCA(n_components=n_components)

    transformer = pca.fit(x)

    return transformer.transform(x)

def run_ica(x, y, name):
    n_components = get_num_ica_comp(x, y, name)
    ica = FastICA(n_components=n_components, whiten=True)

    transformer = ica.fit(x)

    new_x = transformer.fit_transform(x)

    kurt = kurtosis(new_x)
    new_x = pd.DataFrame(data=new_x)

    print ("Kurtosis")
    print(kurt)

    print(new_x.shape)
    drop_cols = []

    for i in range(0, len(kurt)):
        k = kurt[i]

        if abs(k) < 1:
            print ("dropping index {} with kurtosis of {}".format(i, k))
            drop_cols.append(i)

    new_x = new_x.drop(new_x.columns[drop_cols], axis=1)
    
    print(new_x.shape)

    return new_x

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