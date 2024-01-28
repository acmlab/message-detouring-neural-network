from datasets import get_data
import networkx as nx
from torch_geometric.utils import to_networkx
import torch
from tqdm import tqdm
from kfactor import get_kfactor

def count_cycle(d):
    G = to_networkx(d)
    cycle_lens = [len(p) for p in nx.simple_cycles(G, length_bound=5) if len(p) > 2]
    return torch.LongTensor(cycle_lens).bincount()
    
# dn = 'exp'
# for dn in ['exp', 'cexp', 'mutag', 'nci1', 'enzymes', 'ptc_fm', 'imdb-binary', 'dd']:
# for dn in ['texas', 'wisconsin', 'cornell', 'actor', 'squirrel', 'chameleon']:
for dn in ['chameleon']:
    data = get_data(dn, 0, root='node_classification_benchmark/data')
    cycles = []
    for d in tqdm(data):
        if get_kfactor(to_networkx(d)) > 2: 
            cycles.append([])
        else:
            cycle = count_cycle(d)
            cycles.append(cycle)
        # if len(cycle) > 0:
        #     cycles
            # max(cycle)
    torch.save(cycles, f'{dn}_cycle_length_list.zip')
