from torch_geometric.datasets import Planetoid, CoraFull, WebKB, Reddit, WikipediaNetwork, Flickr, TUDataset, Actor#, HeterophilousGraphDataset, BA2MotifDataset#, ZINC, AQSOL, WikiCS, GNNBenchmarkDataset, Amazon, Coauthor
import json, torch, os, random
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_undirected
import networkx as nx
import pickle

random.seed(142857)

def get_data(name, splitid, root='../data', splitn=10):
    path = f'{root}/{name}' 
    os.makedirs(path, exist_ok=True)
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
    elif name in ["collab", "imdb-binary", "imdb-multi", "dd", "enzymes", "proteins", "nci1", "mutag", "ptc_fm", "syntheticnew", "synthetic"]:
        graph = TUDataset(root=path, name=name.upper(), cleaned=True)
    elif name in ['ba2motif']:
        graph = BA2MotifDataset(root=path)
    elif name in ['exp', 'cexp']:
        graph = PlanarSATPairsDataset(root=path)
    elif name in ['graph8c']:
        graph = Grapg8cDataset(root=path)
    elif name in ['sr25']:
        graph = SRDataset(root=path)
    elif name in ['actor']:
        graph = Actor(root=path)
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
        graph = [graph]

    else:
        try:
            splits = json.load(open(f"{path}/{name.upper()}_splits.json", 'r'))
            train_mask = torch.zeros(len(graph)).bool()
            val_mask = torch.zeros(len(graph)).bool()
            test_mask = torch.zeros(len(graph)).bool()
            split = splits[splitid]
            train_mask[split['model_selection'][0]['train']] = True
            val_mask[split['model_selection'][0]['validation']] = True
            test_mask[split['test']] = True
        except:
            train_mask = torch.zeros(len(graph)).bool()
            val_mask = torch.zeros(len(graph)).bool()
            test_mask = torch.zeros(len(graph)).bool()
            ids = torch.arange(len(graph)).tolist()
            random.shuffle(ids)
            splitlen = int(len(graph)/splitn)
            i = splitid
            if i < splitn-1:
                msid = ids[:i*splitlen] + ids[(i+1)*splitlen:] 
                testid = ids[i*splitlen:(i+1)*splitlen] 
            else:
                msid = ids[:i*splitlen]
                testid = ids[i*splitlen:]
            trainid = msid[len(testid):]
            valid = msid[:len(testid)]
            train_mask[trainid] = True
            val_mask[valid] = True
            test_mask[testid] = True

        if not hasattr(graph[0], 'x'):
            for gi in range(len(graph)):
                graph[gi].x = torch.zeros(graph[gi].num_nodes, 1) + torch_geometric.utils.degree(graph[gi].edge_index[0], graph[gi].num_nodes)[:, None]
            

        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask
    return graph



class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["GRAPHSAT.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass  

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/GRAPHSAT.pkl"), "rb"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class Grapg8cDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Grapg8cDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["graph8c.g6"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]   
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i,datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  #sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]   
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i,datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
