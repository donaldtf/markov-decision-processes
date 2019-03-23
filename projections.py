import math
from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

def get_n_components(x):
    n_comps = math.floor(len(x.iloc[0]) / 2)
    print ("Number of components: " + str(n_comps))

    return n_comps

def run_pca(x, _y):
    n_components = get_n_components(x)
    pca = PCA(n_components=n_components, random_state=99)

    transformer = pca.fit(x)

    return transformer.transform(x)

def run_ica(x, _y):
    n_components = get_n_components(x)
    ica = FastICA(n_components=n_components, random_state=99)

    transformer = ica.fit(x)

    return transformer.fit_transform(x)

def run_rp(x, _y):
    n_components = get_n_components(x)
    transformer = random_projection.GaussianRandomProjection(n_components=n_components)

    return transformer.fit_transform(x)

# https://scikit-learn.org/stable/modules/feature_selection.html
def run_tree_selection(x, y):
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(x, y)

    transformer = SelectFromModel(clf, prefit=True)
    
    return transformer.transform(x)