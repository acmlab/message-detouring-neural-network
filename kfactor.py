from tqdm import tqdm 
from datasets import get_data
import networkx as nx
from torch_geometric.utils import to_networkx
import torch

def get_kfactor(G):
    kf = 2
    while True:
        try:
            g=nx.k_factor(G.to_undirected(), kf)
            kf += 1
        except:
            break   
    return kf - 1 

if __name__ == "__main__":
    name = 'graph8c'
    d = get_data(name, 0)
    kf = []
    for di in tqdm(d):
        kf.append(get_kfactor(to_networkx(di)))

    print(kf)
    print(torch.LongTensor(kf).bincount())
