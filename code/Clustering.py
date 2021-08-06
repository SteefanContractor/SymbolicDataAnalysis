import sys
import pickle
import random
import math
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
import MallowsDistance as md

def generate_data(I = 100, k = random.uniform(0, 8)):
    Xi = []
    Yi = []
    ci = []
    ri = []
    for i in range(I):
        ci.append(random.random())
        ri.append(math.exp(random.random()))
        li = ci[-1] - ri[-1]
        mi = ci[-1] + ri[-1]
        Xi.append([li, mi])
        Yi.append([li + k, mi + k])
    ci_dash = [e + k for e in ci]
    return Xi, Yi, ci, ri, ci_dash, k 

def clustering(k=random.uniform(0, 5), scenario="XunifYunif", method=1, a1=1., b1=1., a2=1., b2=1., algorithm="Agglomerative"):
    allowed_algos = ["Agglomerative", "KMedoids"]
    assert algorithm in allowed_algos, "algorithm must be one of Agglomerative or KMedoids."
    Xi, Yi, c1, r1, c2, k = generate_data(k=k)
    r2 = r1
    D_M = md.MallowsDistMatrix(c1, r1, c2, r2, scenario=scenario, method=method, a1=a1, b1=b1, a2=a2, b2=b2)
    if algorithm == "Agglomerative":
        clustering = AgglomerativeClustering(n_clusters=2, affinity="precomputed", linkage="single").fit(D_M)
    else:
        clustering = KMedoids(n_clusters=2, metric="precomputed").fit(D_M)
    pred = clustering.labels_
    return (np.size(np.where(pred[:100] == pred[0])) + np.size(np.where(pred[100:] != pred[0])))/200.
    
def clustering_sim(K, scenario="XunifYunif", method=1, a1=1., b1=1., a2=1., b2=1., rep=10, algorithm="Agglomerative"):
    correct_k = []
    for k in K:        
        correct = []
        for r in range(rep):
            correct.append(clustering(k, scenario=scenario, method=method, a1=a1, b1=b1, a2=a2, b2=b2, algorithm=algorithm))
        correct_k.append(np.mean(correct))
    return correct_k

if __name__ == "__main__":
    
    args = sys.argv
    scenario = args[1]
    method = int(args[2])
    part = int(args[3])
    
    if (scenario in ['XunifYskew', 'XsymYskew', 'XskewYskew']):
        a1, b1, a2, b2 = 1., 5., 1., 5.
    else:
        a1, b1, a2, b2 = 1., 1., 1., 1.
    
    K = np.linspace(0.5, 2, 100, endpoint=True)
    if part == 1:
        K = K[:50]
    else:
        K = K[50:]
        
    correct = clustering_sim(K, scenario=scenario, method=method, a1=a1, b1=b1, a2=a2, b2=b2, rep=10, algorithm="Agglomerative")
    
    # save output
    with open(f"../data/correct_{scenario}_{method}_{part}.pickle", 'wb') as pickleout:
        pickle.dump(correct, pickleout)
        
    
    







