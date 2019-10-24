import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import math
import hdbscan
import umap
import time
from tqdm import tqdm
from QuickshiftPP import *
# add the parent-parent path for quick shift
sys.path.insert(0,'../..')
from MedoidShift_and_QuickShift.quick_shift_ import QuickShift
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

seed_lst = [0, 23, 42, 1234, 43210, 1133557799, 22446688, 123456789, 987654321, 86420]

def find_best_hdbscan(data, y, fname, minclst_range, minsamp_range):
    perf_adjusted_rand_score = []
    perf_adjusted_mutual_info_score = []
    for minclst in tqdm(minclst_range):
        for k in tqdm(minsamp_range):
            clusterer = hdbscan.HDBSCAN(min_cluster_size=int(minclst), min_samples=int(k))
            y_hat = clusterer.fit_predict(data)
            perf_adjusted_rand_score.append((minclst, k, adjusted_rand_score(y_hat, y)))
            perf_adjusted_mutual_info_score.append((minclst, k, adjusted_mutual_info_score(y_hat, y))) 

    minclst, k, score_ari = list(zip(*perf_adjusted_rand_score))
    minclst, k, score_nmi = list(zip(*perf_adjusted_mutual_info_score))
    best_idx_ari = np.argmax(score_ari)
    best_idx_nmi = np.argmax(score_nmi)
    print('Best ARI: %.2f at k: %d and min-clust-size: %d'%(score_ari[best_idx_ari], k[best_idx_ari], minclst[best_idx_ari]))
    print('Best NMI: %.2f at k: %d and min-clust-size: %d'%(score_nmi[best_idx_nmi], k[best_idx_nmi], minclst[best_idx_nmi]))

    best_min_cluster_size = minclst[best_idx_nmi]
    best_min_sample_size = k[best_idx_nmi]
    best_min_mode_size = best_min_cluster_size

    best_k_idx = np.where(np.array(k)==best_min_sample_size)[0]
    #fig, ax = plt.subplots( nrows=1, ncols=1 )
    #plt.plot(np.array(minclst)[best_k_idx], np.array(score_nmi)[best_k_idx], 'b.-',
    #         np.array(minclst)[best_k_idx], np.array(score_ari)[best_k_idx], 'r.-')
    #fig.tight_layout()
    #fig.savefig(fname+'_hdbscan.png', dpi=200, bbox_inches='tight')
    with open(fname+'_hdbscan.pkl', 'wb') as output_file:
        data2dump = {'minclst': minclst, 'k': k, 'ari': score_ari, 'ami': score_nmi}
        pickle.dump(data2dump, output_file)
    #plt.close(fig) 
    
    return best_min_cluster_size, best_min_sample_size

def find_best_qshiftpp(data, y, fname, minsamp_range):
    perf_adjusted_rand_score = []
    perf_adjusted_mutual_info_score = []
    for ib in tqdm(range(9)):
        b = ib * 0.1 + 0.1
        for k in tqdm(minsamp_range):
            model = QuickshiftPP(k=k, beta=b)
            model.fit(data)
            y_hat = model.memberships
            perf_adjusted_rand_score.append((b, k, adjusted_rand_score(y_hat, y)))
            perf_adjusted_mutual_info_score.append((b, k, adjusted_mutual_info_score(y_hat, y))) 

    b, k, score_ari = list(zip(*perf_adjusted_rand_score))
    b, k, score_nmi = list(zip(*perf_adjusted_mutual_info_score))
    best_idx_ari = np.argmax(score_ari)
    best_idx_nmi = np.argmax(score_nmi)
    print('Best ARI: %.2f at k: %d and beta: %.1f'%(score_ari[best_idx_ari], k[best_idx_ari], b[best_idx_ari]))
    print('Best NMI: %.2f at k: %d and beta: %.1f'%(score_nmi[best_idx_nmi], k[best_idx_nmi], b[best_idx_nmi]))
    
    best_k_idx = np.where(np.array(k)==k[best_idx_nmi])[0]
    #fig, ax = plt.subplots( nrows=1, ncols=1 )
    #plt.plot(np.array(b)[best_k_idx], np.array(score_nmi)[best_k_idx], 'b.-',
    #         np.array(b)[best_k_idx], np.array(score_ari)[best_k_idx], 'r.-')
    #fig.tight_layout()
    #fig.savefig(fname+'_qshiftpp.png', dpi=200, bbox_inches='tight')
    with open(fname+'_qshiftpp.pkl', 'wb') as output_file:
        data2dump = {'b': b, 'k': k, 'ari': score_ari, 'ami': score_nmi}
        pickle.dump(data2dump, output_file)
    #plt.close(fig)
    
    k = k[best_idx_nmi]
    b = b[best_idx_nmi]
    return k, b

def find_best_kmeans(data, y, fname):
    #K-Means
    perf_adjusted_rand_score = []
    perf_adjusted_mutual_info_score = []
    for k in tqdm(range(1,30)):
        y_hat = KMeans(n_clusters=k, random_state=seed_lst[0]).fit_predict(data)
        perf_adjusted_rand_score.append((k, adjusted_rand_score(y_hat, y)))
        perf_adjusted_mutual_info_score.append((k, adjusted_mutual_info_score(y_hat, y))) 

    k, score_ari = list(zip(*perf_adjusted_rand_score))
    k, score_nmi = list(zip(*perf_adjusted_mutual_info_score))
    best_idx_ari = np.argmax(score_ari)
    best_idx_nmi = np.argmax(score_nmi)
    print('Best ARI: %.2f at k: %d'%(score_ari[best_idx_ari], k[best_idx_ari]))
    print('Best NMI: %.2f at k: %d'%(score_nmi[best_idx_nmi], k[best_idx_nmi]))

    #best_k_idx = np.where(np.array(k)==k[best_idx_nmi])[0]
    #fig, ax = plt.subplots( nrows=1, ncols=1 )
    #plt.plot(np.array(k), np.array(score_nmi), 'b.-',
    #         np.array(k), np.array(score_ari), 'r.-')
    #fig.tight_layout()
    #fig.savefig(fname+'_kmeans.png', dpi=200, bbox_inches='tight')
    with open(fname+'_kmeans.pkl', 'wb') as output_file:
        data2dump = {'k': k, 'ari': score_ari, 'ami': score_nmi}
        pickle.dump(data2dump, output_file)  
    #plt.close(fig)

    k = k[best_idx_nmi]
    return k

def find_best_qshift(data, y, fname):
    #Quick-Shift
    perf_adjusted_rand_score = []
    perf_adjusted_mutual_info_score = []
    for k in tqdm(range(1,40)):
        bw = k/40
        quick_s_norm = QuickShift(window_type="normal", bandwidth=bw)
        quick_s_norm.fit(data)
        y_hat = quick_s_norm.labels_.astype(np.int)
        perf_adjusted_rand_score.append((bw, adjusted_rand_score(y_hat, y)))
        perf_adjusted_mutual_info_score.append((bw, adjusted_mutual_info_score(y_hat, y))) 

    bw, score_ari = list(zip(*perf_adjusted_rand_score))
    bw, score_nmi = list(zip(*perf_adjusted_mutual_info_score))
    best_idx_ari = np.argmax(score_ari)
    best_idx_nmi = np.argmax(score_nmi)
    print('Best ARI: %.2f at bw: %.2f'%(score_ari[best_idx_ari], bw[best_idx_ari]))
    print('Best NMI: %.2f at bw: %.2f'%(score_nmi[best_idx_nmi], bw[best_idx_nmi]))

    best_bw_idx = np.where(np.array(bw)==bw[best_idx_nmi])[0]
    #fig, ax = plt.subplots( nrows=1, ncols=1 )
    #plt.plot(np.array(bw), np.array(score_nmi), 'b.-',
    #         np.array(bw), np.array(score_ari), 'r.-')
    #fig.tight_layout()
    #fig.savefig(fname+'_qshift.png', dpi=200, bbox_inches='tight')
    with open(fname+'_qshift.pkl', 'wb') as output_file:
        data2dump = {'bw': bw, 'ari': score_ari, 'ami': score_nmi}
        pickle.dump(data2dump, output_file)  
    #plt.close(fig)

    bw = bw[best_idx_nmi]
    return bw

def reduce_dims(data, y, seed_idx):
    if data.shape[1] > 4: 
        X = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=4, random_state=seed_lst[seed_idx]).fit_transform(data)
    else:
        X = data
    print('X.shape: ', X.shape, 'y.shape: ', y.shape)
    return X

def save_many_runs(score_ari, score_nmi, n_clusters, fname, method):
    #fig, ax1 = plt.subplots() 
    #t = np.arange(1, 11)
    #ax1.plot(t, score_nmi, 'b.-',
    #         t, score_ari, 'r.-')
    #ax1.tick_params('y', colors='b')
    #ax1.set_xlabel('Run [index]')
    #ax2 = ax1.twinx()
    #ax2.plot(t, n_clusters, 'g.-')
    #ax2.tick_params('y', colors='g')
    #ax2.set_ylabel('number of clusters', color='g')
    #fig.tight_layout()  
    #fig.savefig(fname+'_'+method+'_many_runs.png', dpi=200, bbox_inches='tight')
    with open(fname+'_'+method+'_many_runs.pkl', 'wb') as output_file:
        data2dump = {'n_clusters': n_clusters, 'ari': score_ari, 'ami': score_nmi}
        pickle.dump(data2dump, output_file) 
    #plt.close(fig)

def main():
    for filename in glob.iglob('**/*.csv', recursive=True):
        print(filename)
        fname = os.path.splitext(os.path.basename(filename))[0]
        print(fname)
        data0 = pd.read_csv(filename, header=None)
        X = data0.iloc[:,:-1].values
        y = data0.iloc[:, -1].values
        num_classes = np.unique(y)[-1] + 1
        num_samples = y.shape[0]

        # find minority class label
        num_samples_minority = math.inf
        for lbl in np.unique(y):
            curr_num_samples = y[y==lbl].shape[0]
            if curr_num_samples < num_samples_minority:
                num_samples_minority = curr_num_samples
                min_label = lbl
        #print(min_label, num_classes, num_samples_minority, num_samples)
        begin = num_samples_minority
        end = (num_samples//num_classes)//1

        minclst_range = np.unique(np.linspace(np.minimum(15, begin), end, num=75, dtype='int32'))
        minsamp_range = np.unique(np.logspace(np.log2(2), np.minimum(np.log2(400), np.log2(num_samples)), base=2, num=75, dtype='int32'))
        print('minsamp_range [k]: ', minsamp_range)
        print('minclst_range    : ', minclst_range)

        data = X
        X = reduce_dims(data, y, 0)

        # K-Means
        print('######K-MEANS######')
        method = 'kmeans'
        n_clusters = []
        score_ari = []
        score_nmi = []
        t0 = time.time()
        k = find_best_kmeans(X, y, fname)
        model_kmeans = KMeans(n_clusters=k, random_state=seed_lst[0])
        y_hat = model_kmeans.fit_predict(X)
        print(time.time()-t0)
        n_clusters.append(np.unique(y_hat)[-1]+1)
        score_ari.append(adjusted_rand_score(y_hat, y))
        score_nmi.append(adjusted_mutual_info_score(y_hat, y))
        print("Adj. Rand Index Score: %f." % score_ari[-1])
        print("Adj. Mutual Info Score: %f." % score_nmi[-1])
        print('Classes: ', np.unique(y_hat), 'n_clusters: ', n_clusters[-1])
        for i in tqdm(range(1, 10)):
            X = reduce_dims(data, y, i)
            y_hat = model_kmeans.fit_predict(X)
            n_clusters.append(np.unique(y_hat)[-1]+1)
            score_ari.append(adjusted_rand_score(y_hat, y))
            score_nmi.append(adjusted_mutual_info_score(y_hat, y))
            print("Adj. Rand Index Score: %f." % score_ari[-1])
            print("Adj. Mutual Info Score: %f." % score_nmi[-1])
            print('Classes: ', np.unique(y_hat), 'n_clusters: ', n_clusters[-1])
        save_many_runs(score_ari, score_nmi, n_clusters, fname, method)

        # Quick-Shift
        print('######QSHIFT######')
        method = 'qshift'
        n_clusters = []
        score_ari = []
        score_nmi = []
        t0 = time.time()
        bw = find_best_qshift(X, y, fname)
        quick_s_norm = QuickShift(window_type="normal", bandwidth=bw)
        quick_s_norm.fit(X)
        y_hat = quick_s_norm.labels_.astype(np.int)
        print(time.time()-t0)
        n_clusters.append(np.unique(y_hat)[-1]+1)
        score_ari.append(adjusted_rand_score(y_hat, y))
        score_nmi.append(adjusted_mutual_info_score(y_hat, y))
        print("Adj. Rand Index Score: %f." % score_ari[-1])
        print("Adj. Mutual Info Score: %f." % score_nmi[-1])
        print('Classes: ', np.unique(y_hat), 'n_clusters: ', n_clusters[-1])
        for i in tqdm(range(1, 10)):
            X = reduce_dims(data, y, i)
            quick_s_norm.fit(X)
            y_hat = quick_s_norm.labels_.astype(np.int)
            n_clusters.append(np.unique(y_hat)[-1]+1)
            score_ari.append(adjusted_rand_score(y_hat, y))
            score_nmi.append(adjusted_mutual_info_score(y_hat, y))
            print("Adj. Rand Index Score: %f." % score_ari[-1])
            print("Adj. Mutual Info Score: %f." % score_nmi[-1])
            print('Classes: ', np.unique(y_hat), 'n_clusters: ', n_clusters[-1])
        save_many_runs(score_ari, score_nmi, n_clusters, fname, method)
 
        # HDBSCAN
        print('######HDBSCAN######')
        method = 'hdbscan'
        n_clusters = []
        score_ari = []
        score_nmi = []
        t0 = time.time()
        best_min_cluster_size, best_min_sample_size = find_best_hdbscan(X, y, fname, minclst_range, minsamp_range)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=int(best_min_cluster_size), min_samples=int(best_min_sample_size))
        y_hat = clusterer.fit_predict(X)
        print(time.time()-t0)
        n_clusters.append(np.unique(y_hat)[-1]+1)
        score_ari.append(adjusted_rand_score(y_hat, y))
        score_nmi.append(adjusted_mutual_info_score(y_hat, y))
        print("Adj. Rand Index Score: %f." % score_ari[-1])
        print("Adj. Mutual Info Score: %f." % score_nmi[-1])
        print('Classes: ', np.unique(y_hat), 'n_clusters: ', n_clusters[-1])
        best_min_cluster_size=17
        if X.shape[0] > 1000:
            best_min_cluster_size = 125
        best_min_sample_size = 3
        clusterer = hdbscan.HDBSCAN(min_cluster_size=int(best_min_cluster_size), min_samples=int(best_min_sample_size))
        for i in tqdm(range(1, 10)):
            X = reduce_dims(data, y, i)
            y_hat = clusterer.fit_predict(X)
            n_clusters.append(np.unique(y_hat)[-1]+1)
            score_ari.append(adjusted_rand_score(y_hat, y))
            score_nmi.append(adjusted_mutual_info_score(y_hat, y))
            print("Adj. Rand Index Score: %f." % score_ari[-1])
            print("Adj. Mutual Info Score: %f." % score_nmi[-1])
            print('Classes: ', np.unique(y_hat), 'n_clusters: ', n_clusters[-1])
            print((y_hat[y_hat==-1].shape[0] / y_hat.shape[0])*100, y_hat[y_hat==-1].shape[0], y_hat.shape[0])
        save_many_runs(score_ari, score_nmi, n_clusters, fname, method)  

        # QSHIFTPP
        print('######QSHIFTPP######')
        method = 'qshiftpp'
        n_clusters = []
        score_ari = []
        score_nmi = []
        t0 = time.time()
        k, b = find_best_qshiftpp(X.copy(order='C'), y, fname, minsamp_range)
        model = QuickshiftPP(k=k, beta=b)
        model.fit(X.copy(order='C'))
        y_hat = model.memberships
        print(time.time()-t0)
        n_clusters.append(np.unique(y_hat)[-1]+1)
        score_ari.append(adjusted_rand_score(y_hat, y))
        score_nmi.append(adjusted_mutual_info_score(y_hat, y))
        print("Adj. Rand Index Score: %f." % score_ari[-1])
        print("Adj. Mutual Info Score: %f." % score_nmi[-1])
        print('Classes: ', np.unique(y_hat), 'n_clusters: ', n_clusters[-1])
        k=25
        if X.shape[0] > 1000:
            k = 60
        model = QuickshiftPP(k=k, beta=.6)
        for i in tqdm(range(1, 10)):
            X = reduce_dims(data, y, i)
            model.fit(X.copy(order='C'))
            y_hat = model.memberships
            n_clusters.append(np.unique(y_hat)[-1]+1)
            score_ari.append(adjusted_rand_score(y_hat, y))
            score_nmi.append(adjusted_mutual_info_score(y_hat, y))
            print("Adj. Rand Index Score: %f." % score_ari[-1])
            print("Adj. Mutual Info Score: %f." % score_nmi[-1])
            print('Classes: ', np.unique(y_hat), 'n_clusters: ', n_clusters[-1])
        save_many_runs(score_ari, score_nmi, n_clusters, fname, method)    
    
if __name__ == '__main__':
    main()
