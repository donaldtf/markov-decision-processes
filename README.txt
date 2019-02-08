##### Machine Learning - OMSCS Spring 2019 ####
##### Author - Donald Ford ####


### Getting Started ###

In order to recreate the results shown in my report, please clone the following repo from github.

https://github.com/donaldtf/ml-supervised-learning

Or you can just use this command to clone it via ssh: `git clone git@github.com:donaldtf/ml-supervised-learning.git`

This repo contains all the code needed to reproduce my results, including the data sets that were used. The project structure looks like this:

/data - this holds the two data sets (hmeq and pulsar_stars) that are used with each algorithm
/images - learning curves for each algorithm are output into this directory
/iteration_curves - this holds the iteration_curves for the algorithms that have an iterative component
/reports - while running each algorithm, I feed the standard output into a report file here instead of to the console.
           This file holds stats on grid search results, test data performance and wall clock times
/utils.py - this is a utility file that holds shared functionality between the algorithms 
            (loading and prepping data, generating learning curves, etc)


### Install Dependencies ###

The code relies on the following dependencies in order to run. You can install them via your favorite method (conda, pip, etc.).

- scikit-learn
- pandas
- numpy
- matplotlib

For example, you could follow these steps to get it working with conda

1. Install Miniconda: https://conda.io/miniconda.html
2. Create a virtual environment and activate it
3. Run the following commands
    - conda install scikit-learn
    - conda install pandas
    - conda install matplotlib
    - conda install numpy

Once these are all installed you should be ready to run the code


### Running the code ###

Running the code is simple once you have your dependencies installed. If you want to run a specific algorithm, simply run:

`python {algorithm}.py` where algorithm is the name of the file you want to run. The available files are

- boosting.py
- decision_tree.py
- knn.py
- neural_network.py
- svm.py

Alternatively, if you would like to run all of the algorithms in bulk you can simply run:

`python run_all.py`

Note: Running all of the algorithms at once may take several minutes (3 - 5 minutes, depending on your machine) to complete.



@ Todo delete this before submitting, also try cloning the repo and running everything one last time
To activate the environment needed run: source activate ml_assignment_1
