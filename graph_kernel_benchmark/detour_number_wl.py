"""
Script makes use of :class:`grakel.WeisfeilerLehman`
"""
from __future__ import print_function
print(__doc__)

import numpy as np

# from sklearn.model_selection import train_test_split

# from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram, ShortestPath

from exp_utils import DATALIST, get_splits, cross_validate_Kfold_SVM, DEN_K
from datetime import datetime
import networkx as nx
from tqdm import trange
import torch
from torch_geometric.datasets import TUDataset
    
def get_degree(edge_index, num_node):
    A = torch.zeros(num_node, num_node)
    A[edge_index[0], edge_index[1]] = 1
    return A.sum(1).long().tolist()

def main():
    for name in DATALIST:
        path = f'../graph_classification_benchmark/data/{name.lower()}' 
        pyg_graph = TUDataset(root=path, name=name.upper())        
        G = [[None, None] for _ in range(len(pyg_graph))]
        y = pyg_graph.y.numpy()
        init_color_fn = f"../graph_classification_benchmark/data/{name.lower()}_{DEN_K}-de_list.zip"
        init_colors = torch.load(init_color_fn)
        # Assign label, it is degree in default
        nid_accumulate = 1
        for gi in trange(len(G), desc=f'{datetime.now()} Init {name} graphs from PyG dataset'):
            g = pyg_graph[gi]
            G[gi][0] = [(ei.item()+nid_accumulate, ej.item()+nid_accumulate) for ei, ej in g.edge_index.T]
            degree_list = get_degree(g.edge_index, g.num_nodes)
            G[gi][1] = {ni+nid_accumulate: d for ni, d in enumerate(degree_list)}
            ############### Use DeN ############################
            assert len(init_colors[gi]) == g.edge_index.shape[1]
            de_list = torch.LongTensor(init_colors[gi]).sort()[0]
            for ni in range(len(G[gi][1])):
                niloc = g.edge_index[0] == ni
                G[gi][1][ni+nid_accumulate] += de_list[niloc].sum().item()*10**(len(str(G[gi][1][ni+nid_accumulate])))
                # for de in de_list[niloc]:
                #     # if de == 0: continue
                #     G[gi][1][ni+nid_accumulate] += de.item()*10**(len(str(G[gi][1][ni+nid_accumulate])))
            ####################################################
            # new_labels = {}
            # label_remap = 1
            # for key in G[gi][1]:
            #     if G[gi][1][key] not in new_labels:
            #         new_labels[G[gi][1][key]] = label_remap
            #         label_remap += 1
            #     G[gi][1][key] = new_labels[G[gi][1][key]]

            nid_accumulate += g.num_nodes
            
        # exit()
        train_splits, val_splits, test_splits = get_splits(G, name=name)#, splitn=10, random_split=True)
        folder = []
        for trainid, valid, testid in zip(train_splits, val_splits, test_splits):
            folder.append([trainid+valid, testid])
        
        Ks = list()
        for i in range(1, 10): 
            gk = WeisfeilerLehman(n_iter=i, base_graph_kernel=VertexHistogram, normalize=False) #VertexHistogram ShortestPath
            K = gk.fit_transform(G)
            Ks.append(K)
        ## CV with Grid search n=10
        accs = cross_validate_Kfold_SVM([Ks], y, n_iter=10, kfolder=folder, random_state=142857)
        print(f"{name} \t Accuracy: {np.mean(accs[0])*100:.2f} (+/- {np.std(accs[0])*100:.2f}) %")



if __name__ == "__main__":
    main()