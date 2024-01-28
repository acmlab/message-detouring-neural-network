from torch_geometric.datasets import Planetoid, CoraFull, WebKB, Reddit, WikipediaNetwork, Flickr, TUDataset, ZINC, AQSOL, WikiCS, GNNBenchmarkDataset, HeterophilousGraphDataset, Amazon, Coauthor, Actor
from torch_geometric.utils import to_networkx, remove_self_loops, add_self_loops, degree
from torch_geometric.data import Data
from torch_geometric.transforms.add_positional_encoding import AddLaplacianEigenvectorPE
import json, torch, random, os
import torch_geometric
import networkx as nx
from tqdm import trange, tqdm
from torch.nn.utils.rnn import pad_sequence
import sys

random.seed(142857)
DEN_K = 2 if len(sys.argv) <= 8 else int(sys.argv[8])
# NAME = ''
PE_K = 10

def get_homo_r(graph):
    homo_n = torch.where(graph.y[graph.edge_index[0]] == graph.y[graph.edge_index[1]])[0].shape[0]
    return homo_n / graph.edge_index.shape[1]
    

def get_data(name, splitn=10, use_trans=False):
    path = f'data/{name}' 
    if name in ['cora', 'pubmed', 'citeseer']:
        graph = Planetoid(root=path, name=name, split='geom-gcn', transform=AddLaplacianEigenvectorPE(PE_K, attr_name='pe'))
    elif name in ['cornell', 'texas', 'wisconsin']:
        graph = WebKB(path ,name=name, transform=AddLaplacianEigenvectorPE(PE_K, attr_name='pe'))
    elif name == 'corafull':
        graph = CoraFull(root=path)
    elif name == 'reddit':
        graph = Reddit(root=path)
    elif name in ["chameleon", "crocodile", "squirrel"]:
        graph = WikipediaNetwork(root=path, name=name, transform=AddLaplacianEigenvectorPE(PE_K, attr_name='pe'))
    elif name == 'flickr':
        graph = Flickr(root=path)
    elif name in ["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]:
        graph = HeterophilousGraphDataset(root=path, name=name)
    elif name in ["collab", "imdb-binary", "imdb-multi", "dd", "enzymes", "proteins", "nci1", "mutag", "ptc_fm"]:
        graph = TUDataset(root=path, name=name.upper(), use_node_attr=True, use_edge_attr=True)
    elif name in ["HCPGender", "HCPActivity", "HCPAge", "HCPFI", "HCPWM"]:
        graph = NeuroGraphDataset(root=path, name=name.upper(), use_node_attr=True, use_edge_attr=True)
    elif name in ["actor"]:
        graph = Actor(root=path, transform=AddLaplacianEigenvectorPE(PE_K, attr_name='pe'))
    if len(graph) == 1:
        graph = graph[0]
        print("Homo. ratio", get_homo_r(graph))
        print("Profile", get_prof(graph))
        if len(graph.train_mask.shape) == 2:
            train_mask = graph.train_mask.T.bool()
            val_mask = graph.val_mask.T.bool()
            test_mask = graph.test_mask.T.bool()
        # elif len(graph.train_mask.shape) == 1:
        #     train_mask = graph.train_mask.bool()
        #     val_mask = graph.val_mask.bool()
        #     test_mask = graph.test_mask.bool()
        if not os.path.exists(f"data/{name}_{DEN_K}-de_list.zip"):
            print('Producing DeN')
            E = graph.edge_index.shape[1]
            G = to_networkx(graph)
            de_list = []
            for j in range(E):
                de_list.append(get_de(G, graph.edge_index[0, j].item(), graph.edge_index[1, j].item(), DEN_K))
            torch.save(de_list, f"data/{name}_{DEN_K}-de_list.zip")
        else:
            de_list = torch.load(f"data/{name}_{DEN_K}-de_list.zip")
            
        if use_trans:
            # maxd = 0
            # xlists, pad_mask_list, edge_indexlist = [], [], []
            # for i, de_list in tqdm(enumerate(de_lists), desc='Build message for detour transformer'):
            dee = torch.FloatTensor(de_list)[:, None]
            if not hasattr(graph, 'x'):
                A = torch.zeros(graph.num_nodes, graph.num_nodes)
                A[graph.edge_index[0], graph.edge_index[1]] = 1
                x=A.sum(1)[:, None]
            else:
                x=graph.x
            # L, V = torch.linalg.eig(A.sum(1)-A)
            # V = V.detach().cpu()
            # pe = V[:, :PE_K].real
            pe = graph.pe
            xlist, pad_mask, edge_index = segment_node_with_neighbor(graph.edge_index, node_attrs=[x, pe], edge_attrs=[dee])
            assert 0 not in edge_index[0].bincount()
            # maxd = edge_index[0].bincount().max().item()
            # xlists.append(xlist)
            # pad_mask_list.append(pad_mask)
            # edge_indexlist.append(edge_index)
        # for i, de_list in enumerate(de_lists):
        # if use_trans:
            # xlist, edge_index, pad_mask = xlists[i], edge_indexlist[i], pad_mask_list[i]
            # pad_mask = torch.cat([pad_mask, torch.zeros(pad_mask.shape[0], maxd-pad_mask.shape[1], 1, dtype=pad_mask.dtype)], 1) 
            # xlist = [torch.cat([s, torch.zeros(s.shape[0], maxd-s.shape[1], s.shape[2], dtype=s.dtype)], 1) for s in xlist]

        if sum(de_list) > 0 and not use_trans:
            de_edge_index = detour_edge(graph.edge_index, de_list, graph.num_nodes)
        else:
            de_edge_index = torch.zeros(2, 0).long()
        A = torch.zeros(graph.num_nodes, graph.num_nodes)
        A[graph.edge_index[0], graph.edge_index[1]] = 1
        if not hasattr(graph, 'x'):
            x=A.sum(1)[:, None]
        else:
            x=graph.x
        if use_trans:
            assert xlist[0].shape[0] == x.shape[0], f"{xlist[0].shape} != {x.shape}, max edge id: {edge_index.max()}, graph node num: {graph.num_nodes}"
            graph_list = [Data(x=x, edge_index=edge_index, node_attr=xlist[0], pe=xlist[1], dee=xlist[2], id=xlist[3], pad_mask=pad_mask, y=graph.y)]
        else:
            graph_list = [Data(x=x, edge_index=graph.edge_index, y=graph.y, de_edge_index=de_edge_index)]


    num_class = len(graph.y.unique())
    assert num_class == graph.y.max()+1, graph.y.unique()
    # return graph_list, torch.where(train_mask)[0].tolist(), torch.where(val_mask)[0].tolist(), torch.where(test_mask)[0].tolist(), num_class
    return graph_list[0], train_mask, val_mask, test_mask, num_class

def segment_node_with_neighbor(edge_index, node_attrs=[], edge_attrs=[], pad_value=0):
    edge_attr_ch = [edge_attr.shape[1] for edge_attr in edge_attrs]
    edge_index, edge_attrs = remove_self_loops(edge_index, torch.cat(edge_attrs, -1) if len(edge_attrs)>0 else None)
    edge_index, edge_attrs = add_self_loops(edge_index, edge_attrs)
    if len(node_attrs[0]) > edge_index.max()+1:
        if edge_attrs is not None:
            edge_attrs = torch.cat([edge_attrs] + [torch.zeros(1, edge_attrs.shape[1]) for i in range(edge_index.max()+1, len(node_attrs[0]))], 0)
        edge_index = torch.cat([edge_index] + [torch.LongTensor([[i, i]]).T for i in range(edge_index.max()+1, len(node_attrs[0]))], 1)

    sortid = edge_index[0].argsort()
    edge_index = edge_index[:, sortid]
    if edge_attrs is not None:
        edge_attrs = edge_attrs[sortid]
    edge_attr_ch = [0] + torch.LongTensor(edge_attr_ch).cumsum(0).tolist()
    edge_attrs = [edge_attrs[:, edge_attr_ch[i]:edge_attr_ch[i+1]] for i in range(len(edge_attr_ch)-1)]
    id_mask = edge_index[0] == edge_index[1]
    edge_attrs.append(id_mask.float()[:, None])
    for i in range(len(node_attrs)):
        node_attrs[i] = torch.cat([node_attrs[i][edge_index[0]], node_attrs[i][edge_index[1]]], -1)
    attrs = node_attrs + edge_attrs
    segment = [torch.where(edge_index[0]==e)[0][0].item() for e in edge_index.unique()] + [edge_index.shape[1]]
    seq = [[] for _ in range(len(attrs))]
    seq_mask = []
    for i in range(len(segment)-1):
        for j in range(len(attrs)):
            attr = attrs[j][segment[i]:segment[i+1]]
            selfloop = torch.where(edge_index[0, segment[i]:segment[i+1]]==edge_index[1, segment[i]:segment[i+1]])[0].item()
            attr = torch.cat([attr[selfloop:selfloop+1], attr[:selfloop], attr[selfloop+1:]]) # Move self loop to the first place
            seq[j].append(attr)
        seq_mask.append(torch.ones(seq[0][i].shape[0], 1))
    seq = [pad_sequence(s, batch_first=True, padding_value=pad_value) for s in seq] # [(N, S, C)]
    seq_mask = pad_sequence(seq_mask, batch_first=True, padding_value=0).float()
    return seq, seq_mask, edge_index


def detour_edge(edge_index, de_list, num_node):
    '''
        edge_index: [2 x E]
        de_list: [E]
    '''
    de_list = torch.FloatTensor(de_list)
    return torch.sparse_coo_tensor(edge_index, de_list, (num_node, num_node)).to_sparse_csr()
    # E = edge_index.shape[1]
    # out = []
    # for i in range(E):
    #     out += [edge_index[:, i] for _ in range(de_list[i])]
    # return torch.stack(out, 1)

def get_de(G, ni, nj, k):
    return len(list(nx.all_simple_paths(G, source=ni, target=nj, cutoff=k))) - 1

def get_prof(G):
    A = torch.zeros(G.num_nodes, G.num_nodes)
    A[G.edge_index[0], G.edge_index[1]] = 1
    x=A.sum(1)
    return f"V: {G.x.shape[0]}, E: {G.edge_index.shape[1]}, avgD: {x.mean().item()}, y: {len(G.y.unique())}"

if __name__ == '__main__':
    get_data(sys.argv[1])