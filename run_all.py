import timeit
import sys
from neural_network import run_nn
from utils import get_hmeq_data, get_pulsar_data, split_data
from projections import run_pca, run_ica, run_rp, run_tree_selection
from cluster import kmeans, gmm, plot_corr
import numpy as np
import pandas as pd

np.random.seed(99)

def run_transform(name, data_x, data_y, transformer):
    print ("Working on {}...".format(name))

    report_name = "reports/{}_nn_output.txt".format(name)
    sys.stdout = open(report_name, "w")

    #2 transform the data
    transform_x = transformer(data_x, data_y, name)
    plot_corr(name, pd.DataFrame(data=transform_x), data_y)
    
    kmeans_name = "{} KMeans Clustered".format(name)
    gmm_name = "{} GMM Clustered".format(name)

    #3 cluster the transformed data
    kmeans_clustered = kmeans(kmeans_name, transform_x, data_y)
    gmm_clustered = gmm(gmm_name, transform_x, data_y)

    #4 run neural network on transformed data
    x_train, x_test, y_train, y_test = split_data(transform_x, data_y)
    run_nn(name, x_train, x_test, y_train, y_test) 
    
    #5 call run_nn on cluster from #3 (clustered from dimensionally reduced)
    kmx_train, kmx_test, kmy_train, kmy_test = split_data(kmeans_clustered, data_y)
    run_nn(kmeans_name, kmx_train, kmx_test, kmy_train, kmy_test)
    
    gmmx_train, gmmx_test, gmmy_train, gmmy_test = split_data(gmm_clustered, data_y)
    run_nn(gmm_name, gmmx_train, gmmx_test, gmmy_train, gmmy_test)

    sys.stdout = sys.__stdout__

    print ("Finished {}!".format(name))
    print()

def run_original(data_x, data_y, name):
    print ("Running {} original charts and clustering".format(name))

    # Run Original
    report_name = "reports/{}_nn_output.txt".format(name)
    sys.stdout = open(report_name, "w")

    x_train, x_test, y_train, y_test = split_data(data_x, data_y)
    run_nn(name, x_train, x_test, y_train, y_test)

    plot_corr("{} Original".format(name), x_train, y_train)

    # Cluster Original (#1)
    kmeans("{} KMeans".format(name), data_x, data_y)
    gmm("{} GMM".format(name), data_x, data_y)

    sys.stdout = sys.__stdout__

    print ("Finished {} original charts and clustering".format(name))

def run_data_set(get_data_fn, name):
    data_x, data_y = get_data_fn()

#     run_original(data_x, data_y, name)

    # Transforms
#     run_transform("{} PCA".format(name), data_x, data_y, run_pca)
#     run_transform("{} ICA".format(name), data_x, data_y, run_ica)
#     run_transform("{} Tree Selection".format(name), data_x, data_y, run_tree_selection)

    for i in range(1, 6):
        run_transform("{} RP Run {}".format(name, i), data_x, data_y, run_rp)

start = timeit.default_timer()

# Pulsar
run_data_set(get_pulsar_data, "Pulsar")

# HMEQ
run_data_set(get_hmeq_data, "HMEQ")

stop = timeit.default_timer()
total_time = stop - start

print ()
print ("FINISHED! Total time taken: " + str(total_time) + " seconds")