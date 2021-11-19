from scipy import linalg
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
"""
A function that takes a list of clusters, and a list of centroids for each cluster, and outputs the N max closest images in each cluster to its centroids
"""
def closest_to_centroid(clusters,centroids,nb_closest=20):
    output = [[] for i in range(len(centroids))]
    #print(clusters)
    for i in range(len(centroids)):
        centroid = centroids[i]
        cluster = clusters[i]
        try :
            cluste_temp = [x.cpu() if x.is_cuda else x for x in cluster]
        except :
            cluste_temp = cluster
        cluster = [list(x) for x in cluste_temp]
        nb_components = 7 if len(cluster)>10 else len(cluster) - 1
        pca = PCA(n_components=nb_components) #args.sty_dim)
        if len(cluster) > nb_closest :
            cluster = pca.fit_transform(cluster)
            centroid = centroid.reshape(1, -1)
            centroid = pca.transform(centroid)
        distances = [linalg.norm(x-centroid) for x in cluster]
        duplicate_distances = distances
        distances.sort()
        if len(distances)>=nb_closest :
            distances = distances[:nb_closest]
        output[i] = [True if x in distances else False for x in duplicate_distances]
    return output



def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    indi = list(ind[0])
    indj = list(ind[1])


    the_sum = sum([w[i, j] for i, j in zip(indi,indj)])
    return the_sum * 1.0 / y_pred.size
