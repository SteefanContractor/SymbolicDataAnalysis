import sys
import pickle
import random
import math
import time
import numpy as np
from numba import jit
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
import MallowsDistance as md

@jit(nopython=True)
def generate_data(I = 20, k = random.uniform(0, 8), csig=1., rlsig=1.):
    Xi = []
    Yi = []
    ci = []
    ri = []
#     random.seed(seed)
    for i in range(I):
        ci.append(random.gauss(0,csig))
        ri.append(random.lognormvariate(0,rlsig))
        li = ci[-1] - ri[-1]
        mi = ci[-1] + ri[-1]
        Xi.append([li, mi])
        Yi.append([li + k, mi + k])
    ci_dash = [e + k for e in ci]
    return Xi, Yi, ci, ri, ci_dash, k  

def clustering(k=random.uniform(0, 5), csig=1., rlsig=1., scenario="XunifYunif", method=1, a1=1., b1=1., a2=1., b2=1., I=20, algorithm="Agglomerative"):
    allowed_algos = ["Agglomerative", "KMedoids"]
    assert algorithm in allowed_algos, "algorithm must be one of Agglomerative or KMedoids."
    Xi, Yi, c1, r1, c2, k = generate_data(I=I, k=k, csig=csig, rlsig=rlsig)
    r2 = r1
    D_M = md.MallowsDistMatrix(c1, r1, c2, r2, scenario=scenario, method=method, a1=a1, b1=b1, a2=a2, b2=b2)
    if algorithm == "Agglomerative":
        clustering = AgglomerativeClustering(n_clusters=2, affinity="precomputed", linkage="single").fit(D_M)
    else:
        clustering = KMedoids(n_clusters=2, metric="precomputed").fit(D_M)
    pred = clustering.labels_
    return (np.size(np.where(pred[:I] == pred[0])) + np.size(np.where(pred[I:] != pred[0])))/(2.*I)
    
def clustering_sim(K, csig=1., rlsig=1., scenario="XunifYunif", method=1, a1=1., b1=1., a2=1., b2=1., I=20, rep=10, algorithm="Agglomerative"):
    correct_k = []
    for k in K:        
        correct = []
        for r in range(rep):
            correct.append(clustering(k, csig=csig, rlsig=rlsig, scenario=scenario, method=method, a1=a1, b1=b1, a2=a2, b2=b2, algorithm=algorithm))
        correct_k.append(np.mean(correct))
    return correct_k

if __name__ == "__main__":
    
    args = sys.argv
    scenario = args[1]
    method = int(args[2])
    
    if (scenario in ['XunifYskew', 'XsymYskew']):
        a1, b1, a2, b2 = 1., 1., 1., 5.
    elif (scenario == 'XskewYskew'):
        a1, b1, a2, b2 = 1., 5., 1., 5.
    else:
        a1, b1, a2, b2 = 1., 1., 1., 1.
    
    tot_start = time.time()
    K = [np.linspace(0.0465,0.93,20), np.linspace(0.155,4.65,30), np.linspace(0.2325,9.3,40)]
    for csig,K in zip([0.1, 0.5, 1.],K):
        print(K)
        for rlsig in [0.1, 0.5, 1.]:
            start = time.time()
            correct = clustering_sim(K, csig=csig, rlsig=rlsig, scenario=scenario, method=method, a1=a1, b1=b1, a2=a2, b2=b2, I=50, rep=30, algorithm="Agglomerative")
            end = time.time()

            print(f'Time taken for csig={csig} and rlsig={rlsig}: {end-start} seconds')

            # save output
            with open(f"../data/correct_csig{csig}_rlsig{rlsig}_{scenario}_{method}.pickle", 'wb') as pickleout:
                pickle.dump(correct, pickleout)
    tot_end = time.time()
    print(f'Total time taken for {scenario}{method}: {(tot_end-tot_start)/60.} minutes')
    
    
    







