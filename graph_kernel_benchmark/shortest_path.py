"""
Script makes use of :class:`grakel.ShortestPath`
"""
from __future__ import print_function
print(__doc__)

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset
from grakel.kernels import ShortestPath, RandomWalk, GraphletSampling

from exp_utils import DATALIST, get_splits, cross_validate_Kfold_SVM
from datetime import datetime
from tqdm import trange
import torch
from torch_geometric.datasets import TUDataset
    
def get_degree(edge_index, num_node):
    A = torch.zeros(num_node, num_node)
    A[edge_index[0], edge_index[1]] = 1
    return A.sum(1).long().tolist()

for name in DATALIST:
    path = f'../graph_classification_benchmark/data/{name.lower()}' 
    pyg_graph = TUDataset(root=path, name=name.upper())        
    G = [[None, None] for _ in range(len(pyg_graph))]
    y = pyg_graph.y.numpy()
    # Assign label, it is degree in default
    nid_accumulate = 1
    for gi in trange(len(G), desc=f'{datetime.now()} Init {name} graphs from PyG dataset'):
        g = pyg_graph[gi]
        G[gi][0] = [(ei.item()+nid_accumulate, ej.item()+nid_accumulate) for ei, ej in g.edge_index.T]
        degree_list = get_degree(g.edge_index, g.num_nodes)
        G[gi][1] = {ni+nid_accumulate: d for ni, d in enumerate(degree_list)}
        # new_labels = {}
        # label_remap = 1
        # for key in G[gi][1]:
        #     if G[gi][1][key] not in new_labels:
        #         new_labels[G[gi][1][key]] = label_remap
        #         label_remap += 1
        #     G[gi][1][key] = new_labels[G[gi][1][key]]
        nid_accumulate += g.num_nodes
    train_splits, val_splits, test_splits = get_splits(G, name=name)
    folder = []
    for trainid, valid, testid in zip(train_splits, val_splits, test_splits):
        folder.append([trainid+valid, testid])
    
    Ks = list()
    # shortest has no hyperparameter normalize=True
    gk = RandomWalk()
    K = gk.fit_transform(G)
    print(K)
    Ks.append(K)

    accs = cross_validate_Kfold_SVM([Ks], y, n_iter=10, kfolder=folder)
    print(f"{name} \t Accuracy: {np.mean(accs[0])*100:.2f} (+/- {np.std(accs[0])*100:.2f}) %")

