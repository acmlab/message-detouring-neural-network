from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv, TransformerConv
from models import ToyMDNN, ToyMPNN

MDNN = ToyMDNN
MPNN = ToyMPNN

MNAMES0 = ['gat']
MNAMES1 = ['gcn', 'gat']
MNAMES2 = ['sage', 'transformer']
MNAMES3 = ['gin']

MFUNC = {
    'gcn': GCNConv,
    'gat': GATConv,
    'sage': SAGEConv,
    'transformer': TransformerConv,
    'gin': GINConv
}