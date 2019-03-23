import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
# Thanks to Scikit-learn
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, validation_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, classification_report
from sklearn import ensemble
import timeit

# Thanks to these sources for examples on loading data in pandas
# https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62
# http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html

# https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star/version/1
def get_pulsar_data():
    pulsar = pd.read_csv('data/pulsar_stars.csv')
    target = "target_class"

    features = list(pulsar.columns.values)
    features.remove(target)

    y = pulsar[target]
    x = pulsar[features]

    return x, y

def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.10, random_state=99)

    return x_train, x_test, y_train, y_test

# https://www.kaggle.com/ajay1735/hmeq-data
def get_hmeq_data():
    hmeq = pd.read_csv('data/hmeq.csv')
    target = "BAD"

    features = list(hmeq.columns.values)
    features.remove(target)
    text_features = ["REASON", "JOB"]

    # Factorize text columns
    for column in text_features:
        factorized_data, not_needed  = pd.factorize(hmeq[column])
        hmeq[column] = factorized_data

    y = hmeq[target]
    x = hmeq[features]

    # https://scikit-learn.org/stable/modules/impute.html
    imp = SimpleImputer(strategy="constant", fill_value=-1)
    x = pd.DataFrame(imp.fit_transform(x))

    return x, y

def compute_stats(y_true, y_pred, name):
    print (name)
    print()
    print ("Final Performance on Test Set")
    mse = mean_squared_error(y_true, y_pred)
    print ("MSE: " + str(mse))
    print()
    
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy Score: " + str(accuracy * 100) + "%")
    print()
    
    f1 = f1_score(y_true, y_pred)
    print("F1 Score: " + str(f1))
    print()

    print(classification_report(y_true, y_pred))
    print()

# This code was taken from the following link
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=-1, scoring="f1", train_sizes=np.linspace(.1, 1.0, 8), file_name="temp.png"):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring, random_state=99)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    plt.savefig(file_name)

def run_optimized(optimized_clf, x_train, y_train, x_test, y_test, name):
    train_start = timeit.default_timer()
    optimized_clf.fit(x_train, y_train)
    train_stop = timeit.default_timer()

    print('Train Time: ', train_stop - train_start)  
    print()

    test_start = timeit.default_timer()
    y_pred = optimized_clf.predict(x_test)
    test_stop = timeit.default_timer()

    print('Test Time: ', test_stop - test_start)  
    print()

    compute_stats(y_test, y_pred, name)