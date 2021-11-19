import numpy as np
from tqdm import trange
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
def kmeans(num_centers, X):
    centers = np.random.uniform(np.min(X[0]), np.max(X[0]), size=(num_centers, X.shape[1]))
    for _ in range(10):
        assigned_to_clusters = np.zeros(len(X))
        for i, x in enumerate(X):
            closest_center = np.argmin([np.linalg.norm(x - center) for center in centers])
            assigned_to_clusters[i] = closest_center
        new_centers = np.array([np.mean(X[assigned_to_clusters == i], axis=0) for i in range(num_centers)])
        convergence = np.sum(np.abs(new_centers - centers))
        centers = new_centers
    print(len(assigned_to_clusters))
    print("\n\n\n")

    return np.array(new_centers), np.array(assigned_to_clusters)

"""
Computing the within sum of squares for a dataset and a number of possible cluster numbers
Args :

Ks : maximal number of clusters to check

input_data : list of input data, but probably np array
"""
def calculate_Wk(Ks, input_data,args):
    e = np.finfo(float).eps
    if(len(input_data) == 2):
        X = input_data[0]
        label = input_data[1]
    else:
        X = input_data
    within_cluster_dists = []
    sil_scores = []
    calinski_scores = []
    for K in range(2,Ks+2):
        dists = 0
        km = KMeans(n_clusters= K, init= 'k-means++', max_iter= 300)
        km.fit(X)
        centroids = km.cluster_centers_
        points = km.labels_
        sil_score = silhouette_score(X, points)
        sil_scores.append(sil_score)
        calinski_score = calinski_harabasz_score(X, points)
        calinski_scores.append(calinski_score)
        for i in range(K):
            cluster_array = X[points == i]
            centroid_dist = []
            dist = 0
            if len(cluster_array) > 0:
                for j in range(len(cluster_array)):
                    centroid_dist.append(np.linalg.norm(centroids[i] - cluster_array[j]))
            dist += np.sum(centroid_dist)
            dists += dist + e
        within_cluster_dists.append(np.log(((dists + e) / K)))
        normalized_wcd = within_cluster_dists - np.max(within_cluster_dists)
    return [normalized_wcd,sil_scores,calinski_scores]

def calculate_normal_Wk(Ks, input_data,args):
    e = np.finfo(float).eps
    if(len(input_data) == 2):
        X = input_data[0]
        label = input_data[1]
    else:
        X = input_data
    within_cluster_dists = []
    for K in range(2,Ks+2):
        dists = 0
        km = KMeans(n_clusters= K, init= 'k-means++', max_iter= 300)
        km.fit(X)
        centroids = km.cluster_centers_
        points = km.labels_
        for i in range(K):
            cluster_array = X[points == i]
            centroid_dist = []
            dist = 0
            if len(cluster_array) > 0:
                for j in range(len(cluster_array)):
                    centroid_dist.append(np.linalg.norm(centroids[i] - cluster_array[j]))
            dist += np.sum(centroid_dist)
            dists += dist + e
        within_cluster_dists.append(np.log(((dists + e) / K)))
        normalized_wcd = within_cluster_dists - np.max(within_cluster_dists)
    return [normalized_wcd]

"""
Getting within sum of squares of an uniform distribution of data
for a specific K, with repeated iterations to get standard deviations
args :

iterations : int, number of iterations in which we test wss of uniform distribution

K :  number of clusters of this simulation

size : tuple containing size of the uniform distribution we want to create*
for example (300,2) for 300 datapoints each with two values
"""

"""
issues : the uniform data generated should have a similar range as real data, not just between 0 and 1
"""
def simulate(K,size,args,minima,maxima,iterations=4):
    e = np.finfo(float).eps
    simulated_Wk = np.zeros((iterations, K)) + e
    for i in range(iterations):
        X = np.random.uniform(minima, maxima, size=size) #(300, 2))
        within_cluster_dists = calculate_normal_Wk(K, X,args)[0]
        simulated_Wk[i] = within_cluster_dists
    Wks = np.mean(simulated_Wk + e, axis=0)
    sks = np.std(simulated_Wk + e, axis=0) * np.sqrt(1 + 1/iterations)
    return Wks, sks

def repeat_clustering(K,data,args,iterations=4):
    e = np.finfo(float).eps
    simulated_Wk = np.zeros((iterations, K)) + e
    for i in range(iterations):
        within_cluster_dists = calculate_normal_Wk(K, data,args)[0]
        simulated_Wk[i] = within_cluster_dists
    Wks = np.mean(simulated_Wk + e, axis=0)
    sks = np.std(simulated_Wk + e, axis=0) * np.sqrt(1 + 1/iterations)
    return Wks, sks

"""
args :


Ks : Maximal number of clusters we could have

input_data : np array  of data inputs

"""
def get_gap_statistics(Ks, input_data,args,run,epoch,type):
    print("computing the gap statistic")
    nb_components = args.sty_dim
    pca = PCA(n_components=nb_components) #args.sty_dim)
    data = pca.fit_transform(input_data)

    var_plot = []
    for i in range(len(pca.explained_variance_ratio_)):
        if len(var_plot) == 0:
            var_plot.append(pca.explained_variance_ratio_[i]*100)
        else :
            var_plot.append(pca.explained_variance_ratio_[i]*100 + var_plot[i-1])
        if var_plot[-1]>90 :
            nb_components = len(var_plot)
            break
    pca = PCA(n_components=nb_components)
    data = pca.fit_transform(input_data)
    print("PCA step done")
    #data = input_data
    """
    # AffinityPropagation
    clustering = AffinityPropagation(random_state=5).fit(data)
    affinity_optimum = len(clustering.cluster_centers_indices_)
    """

    print('Affinity propagation done')
    minima = min(np.amin(data, axis=0))
    maxima = max(np.amax(data, axis=0))
    size = data.shape
    print("beginning simulation")
    Wks, sks = simulate(Ks,size,args,minima,maxima)
    print("beginning gap and sil and calinski")
    within_cluster_dists = calculate_Wk(Ks, data,args)
    sil_scores = within_cluster_dists[1]
    calinski_scores = within_cluster_dists[2]
    within_cluster_dists = within_cluster_dists[0]
    print("begining gap plus")
    real_Wks, real_sks = repeat_clustering(Ks,data,args,)
    G =  Wks - within_cluster_dists
    G_plus = Wks - real_Wks

    print("Statistics computing done")
    initial_k = 2 #args.output_k
    if args.nept :
        for i in range(initial_k):
            run["gap/" + type + "/epoch_" + str(epoch) + '/G'].log(0)
        for i in range(len(G)):
            run['gap/'+type+'/epoch_' + str(epoch) + '/G'].log(G[i])
        for i in range(initial_k):
            run["gap/" + type + "/epoch_" + str(epoch) + '/G_plus'].log(0)
        for i in range(len(G)):
            run['gap/'+type+'/epoch_' + str(epoch) + '/G_plus'].log(G_plus[i])



    # Original gap
    optimum = 0
    for i in range(0, len(G) - 1):
        if(G[i] > G[i+1] - sks[i+1]):
            optimum = i + initial_k
            break
    if optimum == 0 :
        optimum = initial_k
    gap_optimum = optimum
    print("Gap optimum found")


    # slope metric
    epsilon = .1
    window = 5
    optimum = 0
    slopes = []
    corrected_G_plus = []
    slopes = np.gradient(G_plus)
    for i in range(0, len(G_plus)):
        corrected_G_plus.append(G_plus[i] - sks[i] - real_sks[i])
    for i in range(0, len(G_plus) - window):
        if slopes[i] < epsilon and slopes[i] >= 0:
            if G_plus[i] >= max(corrected_G_plus[i+1:i+window]) or max(slopes[i+1:i+window]) <= 0.5 * max(slopes[0:i+1]):
                optimum = i + initial_k +1
                break
    if optimum == 0 :
        optimum = initial_k
    slope_optimum = optimum
    print("slope optimum found")

    # Silhouette score
    sil_optimum = sil_scores.index(max(sil_scores)) + initial_k
    # calinski_harabasz_score
    calinski_optimum = calinski_scores.index(max(calinski_scores)) + initial_k

    """

    # Improved gap v1
    optimum = 0
    optimums = []
    optimal_Gs = []
    for i in range(0, len(G) - 1):
        if(G[i] > G[i+1] - sks[i+1]):
            optimum = i + initial_k
            optimums.append(optimum)
            optimal_Gs.append(G[i])

    if len(optimums) == 0:
        optimum =  initial_k
    else :
        optimum = optimums[optimal_Gs.index(max(optimal_Gs))]

    v1_optimum = optimum

    # Improved gap v 2
    optimum =  initial_k
    if len(optimums) > 0 :
        mean_gap = np.mean(optimal_Gs)
        std_gap = np.std(optimal_Gs)
        for i in range(0, len(optimal_Gs) - 1):
            if optimal_Gs[i]>= mean_gap - std_gap and optimal_Gs[i]<mean_gap+std_gap :
                optimum = optimums[i]
                break
    v2_optimum = optimum
    """
    affinity_optimum = 0
    if args.nept :
        run['gap/'+type+ '_gap'].log(gap_optimum)
        run['gap/'+type+ '_slope'].log(slope_optimum)
        run['gap/'+type+ '_affinity_propagation'].log(affinity_optimum)
        run['gap/'+type+ '_silhouette'].log(sil_optimum)
        run['gap/'+type+ '_calinski'].log(calinski_optimum)

    return [gap_optimum,slope_optimum,affinity_optimum,sil_optimum,calinski_optimum]
