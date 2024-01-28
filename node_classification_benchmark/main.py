from exp_utils import get_data, PE_K
from models import ToyMDNN
import torch
from tqdm import trange, tqdm
import numpy as np
from torch_geometric.loader import DataLoader, NeighborLoader
import sys
from config import *
from datetime import datetime

# dname = 'cora'
# device = 'cuda:0'
# use_skip = True
# use_trans = True
# WO_DeE = False
# # mnames = ['sage', 'transformer']
# mnames = ['gcn', 'gat']
# # mnames = ['gin']
# NN = MDNN
dname = sys.argv[1]
device = sys.argv[2]
use_skip = sys.argv[3]=="1"
if sys.argv[4] == 'gcn':
    mnames = MNAMES1
elif sys.argv[4] == 'sage':
    mnames = MNAMES2
elif sys.argv[4] == 'gin':
    mnames = MNAMES3
elif sys.argv[4] == 'gat':
    mnames = MNAMES0
if sys.argv[5] == 'mdnn':
    NN = MDNN
elif sys.argv[5] == 'mpnn':
    NN = MPNN
use_trans = sys.argv[6]=="1"
WO_DeE = sys.argv[7]=="1"
if use_trans:
    lr = 1e-5
else:
    lr = 1e-5
splits = 10
# batch_size = 256
if dname == 'dd':
    batch_size = 2
elif dname == 'imdb-binary':
    batch_size = 8
else:
    batch_size = 256
epoch = 500
max_patience = 250
verb_inter = 10
hidch = 768
torch.manual_seed(142857)
loader_kwargs = {'batch_size': batch_size, 'num_workers': 6, 'persistent_workers': False}
num_neighbors = [15, 10]
# num_neighbors = [0]

def main_single_graph(NN, CONV_OP, nlayer):
    data, train_masks, val_masks, test_masks, num_class = get_data(dname, use_trans=use_trans)#.to(device)

    accs = []
    for spliti in trange(splits):
        train_mask = train_masks[spliti]
        val_mask = val_masks[spliti]
        test_mask = test_masks[spliti]
        patience = 0
        if use_trans:
            trainloader = NeighborLoader(data, input_nodes=train_mask, num_neighbors=num_neighbors, shuffle=True, **loader_kwargs)
            valloader = NeighborLoader(data, input_nodes=val_mask, num_neighbors=num_neighbors, shuffle=False, **loader_kwargs)
            testloader = NeighborLoader(data, input_nodes=test_mask, num_neighbors=num_neighbors, shuffle=False, **loader_kwargs)

        inch, outch = data.x.shape[1], num_class
        is_graph_level = data.x.shape[0] != data.y.shape[0]

        gnn = NN(CONV_OP, nlayer, inch, outch, hidch, is_graph_level, pedim=PE_K*2, wo_dee=WO_DeE).to(device)
        optimizer = torch.optim.AdamW(gnn.parameters(), lr=lr)
        lr_scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer, num_warmup_steps=None, num_steps=epoch, warmup_proportion=0.005)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400])
        loss_fn = torch.nn.CrossEntropyLoss()
        best_val = 0
        for e in range(epoch):
            gnn.train()
            if use_trans:
                train_loss, train_correct_num = 0, 0
                for graph in trainloader:
                    graph = graph.to(device)
                    logits = gnn(graph.x, graph.edge_index, de_x=[graph.node_attr[:graph.batch_size], graph.pe[:graph.batch_size], graph.dee[:graph.batch_size], graph.id[:graph.batch_size], graph.pad_mask[:graph.batch_size], graph.batch_size], batch=graph.batch, skip_connect=use_skip)
                    loss = loss_fn(input=logits[:graph.batch_size], target=graph.y[:graph.batch_size])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    train_loss += loss.detach().cpu().item()
                    train_correct_num += (logits[:graph.batch_size].argmax(1)==graph.y[:graph.batch_size]).detach().cpu().sum()
            else:
                graph = data.to(device)
                logits = gnn(graph.x, graph.edge_index, de_edge_index0=graph.de_edge_index, batch=graph.batch, skip_connect=use_skip)
                loss = loss_fn(input=logits[train_mask], target=graph.y[train_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss = loss.detach().cpu().item()
                train_correct_num = (logits[train_mask].argmax(1)==graph.y[train_mask]).detach().cpu().sum()
            train_acc = (train_correct_num / sum(train_mask)).item()
            gnn.eval()
            with torch.no_grad():
                if use_trans:
                    val_loss, val_correct_num = 0, 0
                    for graph in valloader:
                        graph = graph.to(device)
                        logits = gnn(graph.x, graph.edge_index, de_x=[graph.node_attr[:graph.batch_size], graph.pe[:graph.batch_size], graph.dee[:graph.batch_size], graph.id[:graph.batch_size], graph.pad_mask[:graph.batch_size], graph.batch_size], batch=graph.batch, skip_connect=use_skip)
                        loss = loss_fn(input=logits[:graph.batch_size], target=graph.y[:graph.batch_size])
                        val_loss += loss.detach().cpu().item()
                        val_correct_num += (logits[:graph.batch_size].argmax(1)==graph.y[:graph.batch_size]).detach().cpu().sum()
                else:
                    graph = data.to(device)
                    logits = gnn(graph.x, graph.edge_index, de_edge_index0=graph.de_edge_index, batch=graph.batch, skip_connect=use_skip)
                    loss = loss_fn(input=logits[val_mask], target=graph.y[val_mask])
                    val_loss = loss.detach().cpu().item()
                    val_correct_num = (logits[val_mask].argmax(1)==graph.y[val_mask]).detach().cpu().sum()
                val_acc = (val_correct_num / sum(val_mask)).item()
                if use_trans:
                    test_loss, test_correct_num = 0, 0
                    for graph in testloader:
                        graph = graph.to(device)
                        logits = gnn(graph.x, graph.edge_index, de_x=[graph.node_attr[:graph.batch_size], graph.pe[:graph.batch_size], graph.dee[:graph.batch_size], graph.id[:graph.batch_size], graph.pad_mask[:graph.batch_size], graph.batch_size], batch=graph.batch, skip_connect=use_skip)
                        loss = loss_fn(input=logits[:graph.batch_size], target=graph.y[:graph.batch_size])
                        test_loss += loss.detach().cpu().item()
                        test_correct_num += (logits[:graph.batch_size].argmax(1)==graph.y[:graph.batch_size]).detach().cpu().sum()
                else:
                    graph = data.to(device)
                    logits = gnn(graph.x, graph.edge_index, de_edge_index0=graph.de_edge_index, batch=graph.batch, skip_connect=use_skip)
                    loss = loss_fn(input=logits[test_mask], target=graph.y[test_mask])
                    test_loss = loss.detach().cpu().item()
                    test_correct_num = (logits[test_mask].argmax(1)==graph.y[test_mask]).detach().cpu().sum()
                test_acc = (test_correct_num / sum(test_mask)).item()
            if e % verb_inter == 0:
                print(f"Split {spliti+1:02d} Epoch {e+1:04d} \t loss:\t({train_loss:.5f}, {val_loss:.5f}, {test_loss:.5f}) \t acc:\t({train_acc*100:.2f}, {val_acc*100:.2f}, {test_acc*100:.2f}) (train, val, test)")
            if val_acc > best_val: 
                best_val = val_acc
                best_acc = test_acc
                patience = 0
            else:
                patience += 1
            if patience > max_patience: break
        accs.append(best_acc)
    return accs


def get_lr_scheduler_with_warmup(optimizer, num_warmup_steps=None, num_steps=None, warmup_proportion=None,
                                 last_step=-1):

    if num_warmup_steps is None and (num_steps is None or warmup_proportion is None):
        raise ValueError('Either num_warmup_steps or num_steps and warmup_proportion should be provided.')

    if num_warmup_steps is None:
        num_warmup_steps = int(num_steps * warmup_proportion)

    def get_lr_multiplier(step):
        if step < num_warmup_steps:
            return (step + 1) / (num_warmup_steps + 1)
        else:
            return 1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=last_step)

    return lr_scheduler


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # xs = [4, 8, 16, 32, 64, 128] 
    # xs = [16, 32, 64, 128]#, 128] 
    # xs = [1, 2, 3, 4, 8]
    xs = [4] 
    # xs = [16] 

    ys = [[] for _ in mnames]   
    yerrs = [[] for _ in mnames]   
    for nlayer in xs:
        print(f"nlayer: {nlayer}")
        for mi, mn in enumerate(mnames):
            accs = main_single_graph(NN, MFUNC[mn], nlayer)
            print(f"ACC: {np.mean(accs)} +/- {np.std(accs)} ({mn})")
            ys[mi].append(np.mean(accs))
            yerrs[mi].append(np.std(accs))

    tag = 'w_skip' if use_skip else 'wo_skip' 
    
    torch.save([mnames, xs, ys, yerrs], f'temp/{dname}-{sys.argv[5]}nlayer vs acc_skip{int(use_skip)}_{datetime.now()}.zip')
    for mi in range(len(mnames)):
        y = ys[mi]
        yerr = yerrs[mi]
        plt.errorbar(xs, y, yerr=yerr, label=mnames[mi])
    plt.legend()
    plt.savefig(f'temp/{dname}-{sys.argv[5]}nlayer vs acc_skip{int(use_skip)}_{datetime.now()}.png')