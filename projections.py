from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection

def run_pca(x):
    pca = PCA(n_components=4, random_state=99)

    transformer = pca.fit(x)

    return transformer.transform(x)

def run_ica(x):
    ica = FastICA(n_components=4, random_state=99)

    transformer = ica.fit(x)

    return transformer.fit_transform(x)

def run_rca(x):
    transformer = random_projection.GaussianRandomProjection(n_components=4, random_state=99)

    return transformer.fit_transform(x)