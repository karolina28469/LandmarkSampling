import sys
import math
from random import sample
import time
import argparse
from numpy import savetxt
from math import log
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial.distance import pdist, squareform
import pylab as pl

# as a parameter 1 == 1D data, 2 = 2D data etc.
# 1D data: index, time p.x, mtd.rbias
# 2D data: index, phi, psi, mtd.rbias
def read_data(dimention):
    if dimention == 1:
        X = np.loadtxt("colvar-1D.data")
    elif dimention == 2:
        X_all = np.loadtxt("colvar-2D.data")
        ds = X_all[:, [2, 3, 51]]
        index = [i for i in range(len(ds))]
        X = np.c_[index, ds]
    else:
        print("No {} dimention data".format(dimention))
        X = []
    return X

def get_rand_sample(X, size):
    indexes = sample(list(X[:, 0]), size)
    sampled_X = np.empty((0, 4), float)
    for i in indexes:
        sampled_X = np.vstack((sampled_X, X[X[:, 0] == i]))
    return sampled_X


#--PLOT--
def plot_hist_1D(X, path):
    plt.hist(X[:, 1], bins=100, alpha=0.5, density=True, label=len(X))

    plt.title('Histogram danych 1D - {} próbek'.format(len(X)), fontsize=16)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('P(x)', fontsize=14)
    plt.xlim([0, 10])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper right', fontsize=14)

    plt.savefig(path + "/(" + time.strftime("%d-%m-%Y %H-%M-%S") + ")result_histogram_1D_" + str(len(X)) + "_probek" + ".pdf")
    plt.show()

def plot_hist_2D(X, path):
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.hist2d(X[:, 1], X[:, 2], bins=(50, 50), cmap=plt.cm.jet, label=len(X))

    plt.title('Histogram danych 2D - {} próbek'.format(len(X)), fontsize=16)
    plt.xlabel('$\phi$', fontsize=14)
    plt.ylabel('$\psi$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    # print(xmin, xmax, ymin, ymax)

    plt.xlim(-3.14157, 3.141542)
    plt.ylim(-3.141582, 3.141293)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(label='logarytm wagi', size=14)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(12)

    plt.savefig(path + "/(" + time.strftime("%d-%m-%Y %H-%M-%S") + ")_result_histogram_2D_" + str(len(X)) + "_probek" + ".pdf")
    plt.show()

def normalize_wages(X):
    log_w = X[:, -1]
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w

def scatter_2d_data(X):
    w = normalize_wages(X)
    fig, ax = plt.subplots()
    ax.set_title('Scatterplot dla zbiioru 2D - znormalizowane wagi', fontsize=16)
    ax.set_xlabel('$\phi$', fontsize=14)
    ax.set_ylabel('$\psi$', fontsize=14)

    s = ax.scatter(X[:, 1], X[:, 2], c=w, s=100)
    ax.set_xlim([np.amin(X[1]), np.amax(X[1])])
    ax.set_ylim([np.amin(X[2]), np.amax(X[2])])
    plt.colorbar(s)
    plt.show()

def plot_voronoi_diagram(X):
    vor = Voronoi(X[:, [1, 2]])
    fig, ax = plt.subplots(figsize=(100, 100))
    ax.set_title('Diagram Voronoi - zbiór 2D', fontsize=16)
    ax.set_xlabel('$\phi$', fontsize=14)
    ax.set_ylabel('$\psi$', fontsize=14)
    ax.set_xlim([np.amin(X[1]), np.amax(X[1])])
    ax.set_ylim([np.amin(X[2]), np.amax(X[2])])
    voronoi_plot_2d(vor, ax, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2, s=100)
    plt.show()

def plot_voronoi_for_clusters(new_X, km):
    v_cluters = Voronoi(km.cluster_centers_[:, 0:2])

    fig, ax = plt.subplots(figsize=(100, 100))
    ax.set_title('Diagram Voronoi dla ({} klastrów) - zbiór 2D'.format(km.n_clusters), fontsize=16)
    ax.set_xlabel('$\phi$', fontsize=14)
    ax.set_ylabel('$\psi$', fontsize=14)
    ax.plot(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 'o', color='k')
    ax.set_xlim([np.amin(new_X[1]), np.amax(new_X[1])])
    ax.set_ylim([np.amin(new_X[2]), np.amax(new_X[2])])
    voronoi_plot_2d(v_cluters, ax)
    plt.show()

def plot_k_means_elbow(X, path, r=30):
    distortions = []
    K = range(1, r)
    for k in K:
        km = KMeans(n_clusters=k)
        km.fit(X[:, 1:3])
        distortions.append(km.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bx-')

    plt.title('Metoda elbow', fontsize=16)
    plt.xlabel('Liczba klastrów', fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel('Wartość funkcji kosztu', fontsize=14)
    plt.yticks(fontsize=12)

    plt.savefig(path + "/(" + time.strftime("%d-%m-%Y %H-%M-%S") + ")_elbow_method" + ".pdf")
    plt.show()

def show_picked_clust(new_X,sampled_clust_ndx):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Wybrane klastry - zbiór 2D', fontsize=16)
    ax.set_xlabel('$\phi$', fontsize=14)
    ax.set_ylabel('$\psi$', fontsize=14)
    # for i in sampled_clust_ndx:
    print(sampled_clust_ndx)
    j = sampled_clust_ndx * 1.0
    plt.scatter(new_X[new_X[:, 4] == j, 1], new_X[new_X[:, 4] == j, 2], label=int(j))
    plt.show()

def overlay_plots(X, sampled_X, lenX, counter, path, lastplot):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 1], X[:, 2], c='b', alpha=0.1)
    plt.scatter(sampled_X[:, 1], sampled_X[:, 2], c='r', alpha=1)

    plt.title('Cały zbiór %d próbek (niebieski) vs %d próbek (czerwony)' % (lenX, len(sampled_X)), fontsize=16)
    plt.xlabel('$\phi$', fontsize=14)
    plt.ylabel('$\psi$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if lastplot == "true":
        plt.savefig(path + "/(" + time.strftime("%d-%m-%Y %H-%M-%S") + ")_results_scatter" + " .pdf")
    else:
        plt.savefig(path + "/(" + time.strftime("%d-%m-%Y %H-%M-%S") + ")_selection_progress_" + str(counter) + ".pdf")
    plt.show()

def plot_KL_divergence(sizeA, KL, path):
    plt.figure(figsize=(10, 6))
    plt.plot(sizeA, KL)
    plt.title('Rozbieżność między całym zbiorem danych - X a jego próbą - A', fontsize=16)
    plt.xlabel('Rozmiar zbioru A', fontsize=14)
    plt.ylabel('KL(X || A)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(path + "/(" + time.strftime("%d-%m-%Y %H-%M-%S") + ")_KL_divergence_2D" + ".pdf")
    plt.show()


#--KMEANS--
def get_centroids(X, labels, K):
    L = math.pi * 2
    centroids = []
    temp = []
    for k in range(K):
        for i in range(len(X)):
            if labels[i] == k:
                if X[i, 0] >= L * 0.5:
                    X[i, 0] -= L
                if X[i, 1] >= L * 0.5:
                    X[i, 1] -= L
                temp.append(X[i])
        centroids.append(np.mean(temp, axis=0))
        temp = []
    return np.array(centroids)

def k_meansk_for_non_periodic_data(X, n_clusters):
    km = KMeans(n_clusters=n_clusters)
    y_pred = km.fit_predict(X[:, 1:3])
    new_X = np.c_[X, y_pred]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Klasteryzacja k-średnich dla {} klastrów - zbiór 2D'.format(n_clusters), fontsize=16)
    ax.set_xlabel('$\phi$', fontsize=14)
    ax.set_ylabel('$\psi$', fontsize=14)
    # plt.show()

    u_labels = np.unique(new_X[:, 4])
    for i in u_labels:
        plt.scatter(new_X[new_X[:, 4] == i, 1], new_X[new_X[:, 4] == i, 2], label=int(i))
        # plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='black', marker='*', s=100)
        plt.annotate(int(i), km.cluster_centers_[int(i)], horizontalalignment='center', verticalalignment='center', size=16, weight='bold')
    plt.show()

    return new_X, km

def k_means_for_periodic_data(X, path, K=5, figures=""):
    L = math.pi * 2
    #find the correct distance matrix
    for d in range(1, 3):
        # all 1-d distances
        pd = pdist(X[:, d].reshape(len(X), 1))
        pd[pd > L * 0.5] -= L
        try:
            total += pd ** 2
        except:
            total = pd ** 2
    # transform the condensed distance matrix...into a square distance matrix
    total = pl.sqrt(total)
    square = squareform(total)

    km2 = KMeans(n_clusters=K).fit(square)
    y_pred = km2.fit_predict(square)
    new_X = np.c_[X, y_pred]

    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 1], X[:, 2], c=km2.labels_, s=50)

    plt.title('Klasteryzacja k-średnich dla k = {}'.format(K), fontsize=16)
    plt.xlabel('$\phi$', fontsize=14)
    plt.ylabel('$\psi$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if figures == 'y':
        plt.savefig(path + "/(" + time.strftime("%d-%m-%Y %H-%M-%S") + ")_kmeans" + ".pdf")
        plt.show()

    return new_X, km2


#--DBSCAN--
def dbscan_for_periodic_data(X, path, threshold = 0.3, figures=""):
    L = math.pi * 2
    for d in range(1,3):
        # all 1-d distances
        pd = pdist(X[:, d].reshape(len(X), 1))
        pd[pd > L * 0.5] -= L
        try:
            total += pd ** 2
        except:
            total = pd ** 2
    # transform the condensed distance matrix...into a square distance matrix
    total = pl.sqrt(total)
    square = squareform(total)

    db2 = DBSCAN(eps=threshold, metric='precomputed').fit(square)
    plt.figure(figsize=(13, 13), dpi=60)
    plt.scatter(X[:, 1], X[:, 2], c=db2.labels_, s=100)

    plt.title('Algorytm DBSCAN, eps=0.1', fontsize=16)
    plt.xlabel('$\phi$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel('$\psi$', fontsize=14)
    plt.yticks(fontsize=12)

    if figures == 'y':
        plt.savefig(path + "/(" + time.strftime("%d-%m-%Y %H-%M-%S") + ")_DBSCAN" + ".pdf")
        plt.show()


#--SELECTION--
def sum_wages_in_custers(X, n_clusters):
    sum_of_clust = np.zeros(n_clusters)
    for i in range(len(X)):
        temp = int(X[i, 4])
        sum_of_clust[temp] += np.exp(X[i, 3])
    return sum_of_clust

def sample_inside_clusters(X, n_clusters, sum_of_clust, pick_n_samples, opt):

    # wybor klastra
    sampled_clust_ndx = sample_clusters(n_clusters, sum_of_clust, 1, 1)
    sampled_clust_ndx = np.unique(X[:, 4])[sampled_clust_ndx]
    print("Wybrano klaser nr: ", sampled_clust_ndx)
    clust_points = X[np.where(X[:, 4] == sampled_clust_ndx)]  # punkty nalezace do wybranego klastra
    print("Ilosc punktów w klastrze ", sampled_clust_ndx, ": ", len(clust_points))

    # wybor punktow z klastra
    logweight_tensor = clust_points[:, 3].tolist()
    ndx = roulette_selection(len(clust_points), logweight_tensor, pick_n_samples, 1, clust_points)

    indexes = []
    indexes.extend(np.unique(ndx))

    sampled_X = np.empty((0, 5), float)
    for i in indexes:
        sampled_X = np.vstack((sampled_X, X[X[:, 0] == i])) #lista indexow wybranych punktów
    s_X = np.unique([tuple(row) for row in sampled_X], axis=0) #gdy wystepuja zdublowane wiersze
    return s_X, indexes

def sample_clusters(n_sample, weights, n_landmark, opt):
    t_weight = sum(weights)
    t_weight = np.power(t_weight, opt)

    running_t_weight = 0
    landmark_indices = []
    selected = np.full(n_sample, False, dtype=bool)

    n_count = 0
    while n_count < n_landmark:
        tw = 0
        r01 = np.random.rand()
        r = (t_weight - running_t_weight) * r01

        for j in range(n_sample):
            if selected[j] == False:
                tw += np.exp(weights[j])
                if r < tw:
                    selected[j] = True
                    landmark_indices.append(j)
                    running_t_weight += weights[j]#np.exp(weights[j])
                    break
        n_count += 1
    return landmark_indices

def roulette_selection(n_sample, logweight_tensor, n_landmark, opt, clust_points):
    temp = np.size(clust_points)

    t_weight = sum(np.exp(logweight_tensor))
    # modified roulette selection //++ t_weight = np.power(t_weight, 0.0001)
    t_weight = np.power(t_weight, opt)

    running_t_weight = 0
    landmark_indices = []
    selected = np.full(n_sample, False, dtype=bool)

    n_count = 0
    while(n_count < n_landmark):
        tw = 0
        r01 = np.random.rand()
        r = (t_weight - running_t_weight) * r01

        for j in range(n_sample):
            if selected[j] == False:
                tw += np.exp(logweight_tensor[j])
                if r < tw:
                    selected[j] = True
                    if temp:
                        landmark_indices.append(clust_points[j, 0])
                    else:
                        landmark_indices.append(j)
                    running_t_weight += np.exp(logweight_tensor[j])
                    break
        n_count += 1
    return landmark_indices


#--KULLBACK–LEIBLER DIVERGENCE
def kl_divergence_1D(HX, HA):
    epsilon = 0.0001
    X = HX + epsilon #whole set
    A = HA + epsilon #sampled set

    s = 0
    d = []
    for i in range(len(X)):
        temp = X[i] * log(X[i]/A[i])
        d.append(temp)
        s += temp
    # plt.plot(d)
    # plt.show()
    return s

def calc_KL_1D(X, A):
    H, bins = np.histogram(X[:, 1], bins=10)
    H = H / H.max()

    H2, bins = np.histogram(A[:, 1], bins=10)
    H2 = H2/H2.max()

    # (X || A)
    kl = kl_divergence_1D(H, H2)
    print('KL(X || (A =', len(A), ') ):', kl)
    return kl

def kl_divergence_2D(HX, HA):
    epsilon = 0.0001
    X = HX + epsilon #whole set
    A = HA + epsilon #sampled set

    s = 0
    d = []
    for i in range(len(X)):
        for j in range(len(X[0])):
            temp = X[i][j] * log(X[i][j]/A[i][j])
            d.append(temp)
            s += temp
    # plt.plot(d)
    # plt.show()
    return s

def calc_KL_2D(X, A):
    H, xedges, yedges = np.histogram2d(X[:, 1], X[:, 2], bins=(100, 100))
    H = H.T
    H = H/max(map(max, H))

    H2, xedges, yedges = np.histogram2d(A[:, 1], A[:, 2], bins=(100, 100))
    H2 = H2.T
    H2 = H2 / max(map(max, H2))

    # (X || A)
    kl = kl_divergence_2D(H, H2)/len(A)
    print('KL(X || (A =', len(A), ') ):', kl)

    return kl

def calc_KL_for_many_sets(X):
    sizeA = []
    KL = []
    for i in range(1, 17):
        print(i, i ** 4)
        A = get_rand_sample(X, i ** 4)

        H, xedges, yedges = np.histogram2d(X[:, 1], X[:, 2], bins=(100, 100))
        H = H.T

        H2, xedges, yedges = np.histogram2d(A[:, 1], A[:, 2], bins=(100, 100))
        H2 = H2.T

        # (X || A)
        kl = kl_divergence_2D(H, H2)
        print('KL(X || A=', len(A), '):', kl)
        sizeA.append(len(A))
        KL.append(kl)
    sizeA.append(len(X))
    KL.append(0.0)

    plt.plot(sizeA, KL)
    plt.show()


def main(argv):
    # Zdefiniowanie argumentów programu
    parser = argparse.ArgumentParser(prog='landmark_sampling.py',
                                    usage='%(prog)s [-options]',
                                    description='DESCRIPTION: Program wybiera reprezentatywną próbkę dla podanego zbioru danych. Program przygotowany w ramach pracy magisterskiej: " Konstrukcja zestawu danych treningowych w uczeniu maszynowym: Landmark sampling". Domyślne wartości: -dataset 2 -size 3000 -n_clusters 10 -n_samples 200 -figures y -path outputs',
                                    epilog='Enjoy the program! :)')

    parser.add_argument('-dataset', default='2', help='opcja 1 lub 2 (1: zbiór jednowymiarowy, 2: zbiór dwuwymiarowy)')
    parser.add_argument('-size', default='3000', help='wielkość próbki')
    parser.add_argument('-n_clusters', default='10', help='ilość klastrów')
    parser.add_argument('-n_samples', default='200', help='maksymalna ilość próbek wybrana za jednym razem z klastra')
    parser.add_argument('-figures', default='y', help='y - tak dla dodatkowych wykresów')
    parser.add_argument('-path', default='outputs', help='scieżka do wykresów')
    parser.add_argument('-opt', default='0.001', help='wartośc wykładnika - optymalizacja selekcji ruletki')
    args = vars(parser.parse_args())

    # Wczytanie argumentów programu
    dataset = args['dataset']
    size = int(args['size'])
    n_clusters = int(args['n_clusters'])
    n_samples = int(args['n_samples'])
    figures = args['figures']
    path = args['path']
    opt = float(args['opt'])
    print("Argumenty programu: ", dataset, size, n_clusters, n_samples, figures, path, opt)

    if(dataset == '1'):
        X = read_data(1)
        if figures == 'y':
            plot_hist_1D(X, path)

        pick_n_samples = size
        ndx = roulette_selection(len(X), X[:, 2], pick_n_samples, opt, np.array([]))
        ndx.sort()
        R = np.empty(shape=[pick_n_samples, 3]) # results array
        for i in range(len(ndx)):
            R[i] = (X[ndx[i]])

        plot_hist_1D(R, path)
        savetxt(path + "/results_1D_" + str(len(R)) + "_probek" + ".csv", R, delimiter='\t')

        kl = calc_KL_1D(X, R)

    elif(dataset == '2'):
        X = read_data(2)

        if figures == 'y':
            plot_hist_2D(X, path)

        A = get_rand_sample(X, 5000)
        X = A

        # -- KMEANS --
        if figures == 'y':
            plot_k_means_elbow(X, path, 30)
        new_X, km = k_means_for_periodic_data(X, path, n_clusters, figures)
        sum_of_clust = sum_wages_in_custers(new_X, n_clusters)

        result = np.empty((0, 5), float)
        counter = 1 # used for counting figures - naming

        sizeA = []
        KL = []

        while len(result) < size:
            print("\n##POZOSTALO ", size - len(result), "\n")
            NSAMPLES = n_samples
            if size - len(result) < n_samples:
                NSAMPLES = size - len(result)
            sum_of_clust = sum_wages_in_custers(new_X, n_clusters)
            sum_of_clust = sum_of_clust[sum_of_clust != 0]

            sampled_X, indexes = sample_inside_clusters(new_X, n_clusters, sum_of_clust, NSAMPLES, opt)
            r = np.vstack((result, sampled_X))

            result = np.unique([tuple(row) for row in r], axis=0)
            d = []
            for s in range(len(new_X)):
                if any((new_X[s] == result[:]).all(1)) == True:
                    d.append(s)
            dd = sorted(d, reverse=True)
            for d in dd:
                new_X = np.delete(new_X, d, 0)

            print("DO TEJ PORY WYBRANO ", len(result), " PROBEK")

            if figures == 'y':
                overlay_plots(new_X, result, len(X), counter, path, "false")

            kl = calc_KL_2D(X, result)
            sizeA.append(len(result))
            KL.append(kl)

            counter += 1

        overlay_plots(X, result, len(X), counter, path, "true")
        savetxt(path + "/results_2D_" + str(len(result)) + "_probek" + ".csv", result, delimiter='\t')

        plot_hist_2D(result, path)

        sizeA.append(len(X))
        KL.append(0.0)
        plot_KL_divergence(sizeA, KL, path)

if __name__ == "__main__":
    main(sys.argv[1:])
