import numpy as np
# Thanks to Scikit-learn
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
from sklearn.neural_network import MLPClassifier
from utils import run_optimized, plot_learning_curve

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

def run_nn(name, x_train, x_test, y_train, y_test):
    img_name = "images/{}_nn_learning_curve.png".format(name)
    img_title = '{} Neural Net Learning Curve'.format(name)
    iter_title = '{} Neural Net Iteration Learning Curve'.format(name)

    optimized_clf = MLPClassifier(max_iter=1000, alpha=0.001, hidden_layer_sizes=11, random_state=99)

    plot_learning_curve(
        optimized_clf,
        title=img_title,
        file_name=img_name,
        X=x_train,
        y=y_train,
        )

    run_optimized(optimized_clf, x_train, y_train, x_test, y_test, name)
