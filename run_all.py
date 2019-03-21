import timeit
from neural_network import run_nn
from utils import get_hmeq_data, get_pulsar_data, split_data
from projections import run_pca, run_ica, run_rca
from kmeans import cluster

def run_transform(name, data_x, data_y, transformer):
    report_name = "reports/{}_nn_output.txt".format(name)
    sys.stdout = open(report_name, "w")

    new_x = transformer(data_x)
    x_train, x_test, y_train, y_test = split_data(new_x, data_y)

    cluster(name, new_x, data_y)

    run_nn(name, x_train, x_test, y_train, y_test)

    sys.stdout = sys.__stdout__


start = timeit.default_timer()

# Pulsar
pulsar_x, pulsar_y = get_pulsar_data()

# Pulsar Original
report_name = "reports/Pulsar_nn_output.txt"
sys.stdout = open(report_name, "w")

x_train, x_test, y_train, y_test = split_data(pulsar_x, pulsar_y)
run_nn("Pulsar", x_train, x_test, y_train, y_test)

sys.stdout = sys.__stdout__

# Cluser Original
cluster("Pulsar", pulsar_x, pulsar_y)

# Pulsar Transforms
run_transform("pulsar_pca", pulsar_x, pulsar_y, run_pca)
run_transform("pulsar_ica", pulsar_x, pulsar_y, run_ica)
run_transform("pulsar_rca", pulsar_x, pulsar_y, run_rca)


# HMEQ
hmeq_x, hmeq_y = get_hmeq_data()

# HMEQ Original
report_name = "reports/HMEQ_nn_output.txt"
sys.stdout = open(report_name, "w")

x_train, x_test, y_train, y_test = split_data(hmeq_x, hmeq_y)
run_nn("HMEQ", x_train, x_test, y_train, y_test)

sys.stdout = sys.__stdout__

# HMEQ Transforms
run_transform("hmeq_pca", hmeq_x, hmeq_y, run_pca)
run_transform("hmeq_ica", hmeq_x, hmeq_y, run_ica)
run_transform("hmeq_rca", hmeq_x, hmeq_y, run_rca)


stop = timeit.default_timer()
total_time = stop - start

print ()
print ("FINISHED! Total time taken: " + str(total_time) + " seconds")