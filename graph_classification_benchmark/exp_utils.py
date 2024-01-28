from torch_geometric.datasets import Planetoid, CoraFull, WebKB, Reddit, WikipediaNetwork, Flickr, TUDataset, ZINC, AQSOL, WikiCS, GNNBenchmarkDataset, HeterophilousGraphDataset, Amazon, Coauthor
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
DEN_K = 2 if len(sys.argv)<=8 else int(sys.argv[8])
NAME = ''

def get_data(name, splitn=10, use_trans=False):
    path = f'data/{name}' 
    if name in ['cora', 'pubmed', 'citeseer']:
        graph = Planetoid(root=path, name=name, split='geom-gcn')
    elif name in ['cornell', 'texas', 'wisconsin']:
        graph = WebKB(path ,name=name)
    elif name == 'corafull':
        graph = CoraFull(root=path)
    elif name == 'reddit':
        graph = Reddit(root=path)
    elif name in ["chameleon", "crocodile", "squirrel"]:
        graph = WikipediaNetwork(root=path, name=name)
    elif name == 'flickr':
        graph = Flickr(root=path)
    elif name in ["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]:
        graph = HeterophilousGraphDataset(root=path, name=name)
    elif name in ["collab", "imdb-binary", "imdb-multi", "dd", "enzymes", "proteins", "nci1", "mutag", "ptc_fm"]:
        graph = TUDataset(root=path, name=name.upper(), use_node_attr=True, use_edge_attr=True)
    if len(graph) == 1:
        graph = graph[0]
        if len(graph.train_mask.shape) == 2:
            train_mask = graph.train_mask.T[splitid].bool()
            val_mask = graph.val_mask.T[splitid].bool()
            test_mask = graph.test_mask.T[splitid].bool()
        elif len(graph.train_mask.shape) == 1:
            train_mask = graph.train_mask.bool()
            val_mask = graph.val_mask.bool()
            test_mask = graph.test_mask.bool()

    else:
        try:
            splits = json.load(open(f"data/{name.upper()}_splits.json", 'r'))
            train_mask = torch.zeros(splitn, len(graph)).bool()
            val_mask = torch.zeros(splitn, len(graph)).bool()
            test_mask = torch.zeros(splitn, len(graph)).bool()
            for splitid in range(splitn):
                split = splits[splitid]
                train_mask[splitid, split['model_selection'][0]['train']] = True
                val_mask[splitid, split['model_selection'][0]['validation']] = True
                test_mask[splitid, split['test']] = True
        except:
            train_mask = torch.zeros(splitn, len(graph)).bool()
            val_mask = torch.zeros(splitn, len(graph)).bool()
            test_mask = torch.zeros(splitn, len(graph)).bool()
            ids = torch.arange(len(graph)).tolist()
            random.shuffle(ids)
            splitlen = int(len(graph)/splitn)
            for i in range(splitn):
                if i < splitn-1:
                    msid = ids[:i*splitlen] + ids[(i+1)*splitlen:] 
                    testid = ids[i*splitlen:(i+1)*splitlen] 
                else:
                    msid = ids[:i*splitlen]
                    testid = ids[i*splitlen:]
                trainid = msid[len(testid):]
                valid = msid[:len(testid)]
                train_mask[i, trainid] = True
                val_mask[i, valid] = True
                test_mask[i, testid] = True
        
        if not os.path.exists(f"data/{name}_{DEN_K}-de_list.zip"):
            de_lists = []
            for i in trange(len(graph), desc='Producing DeN'):
                E = graph[i].edge_index.shape[1]
                G = to_networkx(graph[i])
                de_list = []
                for j in range(E):
                    de_list.append(get_de(G, graph[i].edge_index[0, j].item(), graph[i].edge_index[1, j].item(), DEN_K))
                de_lists.append(de_list)
            torch.save(de_lists, f"data/{name}_{DEN_K}-de_list.zip")
        else:
            de_lists = torch.load(f"data/{name}_{DEN_K}-de_list.zip")
        graph_list = []
        if use_trans:
            PE_K = min([d.num_nodes for d in graph])
            maxd = 0
            xlists, pad_mask_list, edge_indexlist = [], [], []
            for i, de_list in tqdm(enumerate(de_lists), desc='Build message for detour transformer'):
                dee = torch.FloatTensor(de_list)[:, None]
                A = torch.zeros(graph[i].num_nodes, graph[i].num_nodes)
                A[graph[i].edge_index[0], graph[i].edge_index[1]] = 1
                if not hasattr(graph, 'x'):
                    x=A.sum(1)[:, None]
                else:
                    x=graph[i].x
                L, V = torch.linalg.eig(A.sum(1)-A)
                pe = V[:, :PE_K].real
                xlist, pad_mask, edge_index = segment_node_with_neighbor(graph[i].edge_index, node_attrs=[x, pe], edge_attrs=[dee])
                assert 0 not in edge_index[0].bincount()
                maxd = max(edge_index[0].bincount().max().item(), maxd)
                xlists.append(xlist)
                pad_mask_list.append(pad_mask)
                edge_indexlist.append(edge_index)
        for i, de_list in enumerate(de_lists):
            if use_trans:
                xlist, edge_index, pad_mask = xlists[i], edge_indexlist[i], pad_mask_list[i]
                pad_mask = torch.cat([pad_mask, torch.zeros(pad_mask.shape[0], maxd-pad_mask.shape[1], 1, dtype=pad_mask.dtype)], 1) 
                xlist = [torch.cat([s, torch.zeros(s.shape[0], maxd-s.shape[1], s.shape[2], dtype=s.dtype)], 1) for s in xlist]
            if sum(de_list) > 0 and not use_trans:
                de_edge_index = detour_edge(graph[i].edge_index, de_list, graph[i].num_nodes)
            elif use_trans:
                de_edge_index = torch.zeros(2, 0).long()
            # else:
            #     continue
            A = torch.zeros(graph[i].num_nodes, graph[i].num_nodes)
            A[graph[i].edge_index[0], graph[i].edge_index[1]] = 1
            if not hasattr(graph, 'x'):
                x=A.sum(1)[:, None]
            else:
                x=graph[i].x
            if use_trans:
                assert xlist[0].shape[0] == x.shape[0], f"{xlist[0].shape} != {x.shape}, max edge id: {edge_index.max()}, graph node num: {graph[i].num_nodes}"
                graph_list.append(Data(x=x, edge_index=edge_index, node_attr=xlist[0], pe=xlist[1], dee=xlist[2], id=xlist[3], pad_mask=pad_mask, y=graph[i].y))
            else:
                # graph_list.append(Data(x=x, edge_index=graph[i].edge_index, y=graph[i].y, de_edge_index=de_edge_index))
                assert len(de_list) == graph[i].edge_index.shape[1], f"{len(de_list)} != {graph[i].edge_index.shape[1]}"
                graph_list.append(Data(x=x, edge_index=graph[i].edge_index, y=graph[i].y, de_edge_index=torch.stack([torch.FloatTensor(de_list), torch.FloatTensor(de_list)])))
                

    # if not hasattr(graph, 'x'):
    #     for gi in range(len(graph)):
    #         graph[gi].x = torch.zeros(graph[gi].num_nodes, 1)
    #         graph[gi].x[:, 0] = torch_geometric.utils.degree(graph[gi].edge_index[0], graph.num_nodes)

    # graph.train_mask = train_mask
    # graph.val_mask = val_mask
    # graph.test_mask = test_mask

    num_class = graph.num_classes
    # return graph_list, torch.where(train_mask)[0].tolist(), torch.where(val_mask)[0].tolist(), torch.where(test_mask)[0].tolist(), num_class
    return graph_list, train_mask, val_mask, test_mask, num_class

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
        seq_mask.append(torch.ones(seq[j][0].shape[0], 1))
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
# def detour_edge(edge_index, de_list):
#     '''
#         edge_index: [2 x E]
#         de_list: [E]
#     '''
#     E = edge_index.shape[1]
#     out = []
#     for i in range(E):
#         out += [edge_index[:, i] for _ in range(de_list[i])]
#     return torch.stack(out, 1)

def get_de(G, ni, nj, k):
    return len(list(nx.all_simple_paths(G, source=ni, target=nj, cutoff=k))) - 1

if __name__ == '__main__':
    get_data(NAME)