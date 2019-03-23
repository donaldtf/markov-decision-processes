##### Machine Learning - OMSCS Spring 2019 ####
##### Author - Donald Ford ####


### Getting Started ###

In order to recreate the results shown in my report, please clone the following repo from github.

https://github.com/donaldtf/ml-unsupervised-learning

Or you can just use this command to clone it via ssh: `git clone git@github.com:donaldtf/ml-unsupervised-learning.git`

This repo contains all the code needed to reproduce my results, including the data sets that were used. The project structure looks like this:

/data - this holds the two data sets (hmeq and pulsar_stars) that are used with each algorithm
/images - learning curves for each algorithm are output into this directory
/elbow_curves - kmeans elbow curves are output to this directory
/reports - while running each algorithm, I feed the standard output into a report file here instead of to the console.
           This file holds stats on grid search results, test data performance and wall clock times
/utils.py - this is a utility file that holds shared functionality between the algorithms 
            (loading and prepping data, plotting learning curves, etc)


### Install Dependencies ###

The code relies on the following dependencies in order to run. You can install them via your favorite method (conda, pip, etc.).

- scikit-learn
- pandas
- numpy
- matplotlib
- scipy
- seaborn
- kneed (directions for installing found here: https://github.com/arvkevi/kneed)

Once these are all installed you should be ready to run the code

### Running the code ###

Running the code is simple once you have your dependencies installed. Simply run the following command

`python run_all.py`

This will generate all of the plots shown in the report, plus other plots that were excluded from the report due to the constraint on report length.

Note: Running all of the algorithms at once may take several minutes (10 - 15 minutes, depending on your machine) to complete.
