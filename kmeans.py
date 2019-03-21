from sklearn.cluster import KMeans
import numpy as np
from utils import get_hmeq_data, get_pulsar_data, compute_stats


def cluster(name, x, y):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(x)

    labels = list(kmeans.labels_)

    print (name)
    print (kmeans.cluster_centers_)

    print ('Data Length: ' + str(len(labels)))
    print ('Cluster 1: ' + str(labels.count(0)))
    print ('Cluster 2: ' + str(labels.count(1)))

    y_pred = kmeans.predict(x)

    compute_stats(y, y_pred)


if __name__ == "__main__":
    print ("Running KMeans")

    pulsar_x, pulsar_y = get_pulsar_data()
    cluster("Pulsar", pulsar_x, pulsar_y)

    hmeq_x, hmeq_y = get_hmeq_data()
    cluster("HMEQ", hmeq_x, hmeq_y)

    print ("Finished Running KMeans")

