import timeit
import sys
from neural_network import run_nn
from utils import get_hmeq_data, get_pulsar_data, split_data
from projections import run_pca, run_ica, run_rca
from kmeans import cluster

def run_transform(name, data_x, data_y, transformer):
    print ("Working on {}...".format(name))

    report_name = "reports/{}_nn_output.txt".format(name)
    sys.stdout = open(report_name, "w")

    new_x = transformer(data_x) # (#2)
    x_train, x_test, y_train, y_test = split_data(new_x, data_y)

    cluster(name, new_x, data_y) # (#3)

    run_nn(name, x_train, x_test, y_train, y_test) # (#4)

    sys.stdout = sys.__stdout__

    print ("Finished {}!".format(name))
    print()

def run_data_set(get_data_fn, name):
    data_x, data_y = get_data_fn()

    # Run Original
    report_name = "reports/{}_nn_output.txt".format(name)
    sys.stdout = open(report_name, "w")

    x_train, x_test, y_train, y_test = split_data(data_x, data_y)
    run_nn(name, x_train, x_test, y_train, y_test)

    sys.stdout = sys.__stdout__

    # Cluster Original (#1)
    cluster(name, data_x, data_y)

    # Transforms
    run_transform("{}_pca".format(name), data_x, data_y, run_pca)
    run_transform("{}_ica".format(name), data_x, data_y, run_ica)
    run_transform("{}_rca".format(name), data_x, data_y, run_rca)


start = timeit.default_timer()

# Pulsar
run_data_set(get_pulsar_data, "Pulsar")

# HMEQ
run_data_set(get_hmeq_data, "HMEQ")

stop = timeit.default_timer()
total_time = stop - start

print ()
print ("FINISHED! Total time taken: " + str(total_time) + " seconds")