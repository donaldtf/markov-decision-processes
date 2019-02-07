from decision_tree import run_pulsar_dt, run_hmeq_dt
from knn import run_hmeq_knn, run_pulsar_knn
from boosting import run_hmeq_boosting, run_pulsar_boosting
from svm import run_hmeq_svm, run_pulsar_svm
from neural_network import run_hmeq_nn, run_pulsar_nn

print ("---- Decision Tree ----")

run_hmeq_dt()
run_pulsar_dt()

print()
print ("---- KNN ----")

run_hmeq_knn()
run_pulsar_knn()

print()
print ("---- Boosting ----")

run_hmeq_boosting()
run_pulsar_boosting()

print()
print ("---- SVM ----")

run_hmeq_svm()
run_pulsar_svm()

print()
print ("---- Neural Network ----")

run_hmeq_nn()
run_pulsar_nn()