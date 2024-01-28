from exp_utils import get_data
from models import ToyMDNN
import torch
from tqdm import trange, tqdm
import numpy as np
from torch_geometric.loader import DataLoader
import sys
from config import *
from datetime import datetime

# dname = 'mutag'
# device = 'cuda:0'
# use_skip = True
# use_trans = True
# WO_DeE = False
# mnames = ['sage', 'transformer']
# mnames = ['gcn', 'gat']
# mnames = ['gin']
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
lr = 1e-5
splits = 10
# batch_size = 256
if dname == 'dd':
    batch_size = 2
elif dname == 'imdb-binary':
    batch_size = 8
else:
    batch_size = 16
epoch = 500
max_patience = 250
verb_inter = 10
hidch = 768
torch.manual_seed(142857)

def main_multi_graph(NN, CONV_OP, nlayer):
    if nlayer >= 64: 
        bs = 4
    else:
        bs = batch_size
    accs = []
    data, train_masks, val_masks, test_masks, num_class = get_data(dname, use_trans=use_trans)#.to(device)
    for spliti in trange(splits):
        train_mask = torch.where(train_masks[spliti])[0].tolist()
        val_mask = torch.where(val_masks[spliti])[0].tolist()
        test_mask = torch.where(test_masks[spliti])[0].tolist() 
        patience = 0
        trainloader = DataLoader([data[i] for i in train_mask], batch_size=bs, shuffle=True)
        valloader = DataLoader([data[i] for i in val_mask], batch_size=bs, shuffle=False)
        testloader = DataLoader([data[i] for i in test_mask], batch_size=bs, shuffle=False)

        inch, outch = data[0].x.shape[1], num_class
        is_graph_level = data[0].x.shape[0] != data[0].y.shape[0]

        gnn = NN(CONV_OP, nlayer, inch, outch, hidch, is_graph_level, pedim=min([d.num_nodes for d in data])*2, wo_dee=WO_DeE).to(device)
        optimizer = torch.optim.AdamW(gnn.parameters(), lr=lr)
        lr_scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer, num_warmup_steps=None, num_steps=epoch, warmup_proportion=0.005)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400])
        loss_fn = torch.nn.CrossEntropyLoss()
        best_val = 0
        for e in range(epoch):
            gnn.train()
            train_loss, train_correct_num = 0, 0
            for graph in trainloader:
            # for graph in tqdm(trainloader, desc=f"Train epoch {e+1}"):
                graph = graph.to(device)
                if not use_trans:
                    logits = gnn(graph.x, graph.edge_index, de_edge_index0=graph.de_edge_index, batch=graph.batch, skip_connect=use_skip)
                else:
                    logits = gnn(graph.x, graph.edge_index, de_x=[graph.node_attr, graph.pe, graph.dee, graph.id, graph.pad_mask], batch=graph.batch, skip_connect=use_skip)
                loss = loss_fn(input=logits, target=graph.y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss += loss.detach().cpu().item()
                train_correct_num += (logits.argmax(1)==graph.y).detach().cpu().sum()
            train_acc = (train_correct_num / len(train_mask)).item()
            gnn.eval()
            with torch.no_grad():
                val_loss, val_correct_num = 0, 0
                for graph in valloader:
                    graph = graph.to(device)
                    if not use_trans:
                        logits = gnn(graph.x, graph.edge_index, de_edge_index0=graph.de_edge_index, batch=graph.batch, skip_connect=use_skip)
                    else:
                        logits = gnn(graph.x, graph.edge_index, de_x=[graph.node_attr, graph.pe, graph.dee, graph.id, graph.pad_mask], batch=graph.batch, skip_connect=use_skip)
                    loss = loss_fn(input=logits, target=graph.y)
                    val_loss += loss.detach().cpu().item()
                    val_correct_num += (logits.argmax(1)==graph.y).detach().cpu().sum()
                val_acc = (val_correct_num / len(val_mask)).item()
                test_loss, test_correct_num = 0, 0
                for graph in testloader:
                    graph = graph.to(device)
                    if not use_trans:
                        logits = gnn(graph.x, graph.edge_index, de_edge_index0=graph.de_edge_index, batch=graph.batch, skip_connect=use_skip)
                    else:
                        logits = gnn(graph.x, graph.edge_index, de_x=[graph.node_attr, graph.pe, graph.dee, graph.id, graph.pad_mask], batch=graph.batch, skip_connect=use_skip)
                    loss = loss_fn(input=logits, target=graph.y)
                    test_loss += loss.detach().cpu().item()
                    test_correct_num += (logits.argmax(1)==graph.y).detach().cpu().sum()
                test_acc = (test_correct_num / len(test_mask)).item()
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

def one_step(model, graph, mask, optimizer, scheduler, loss_fn, istrain=False):
    logits = model(graph.x, graph.edge_index)
    loss = loss_fn(input=logits[mask], target=graph.y[mask])
    if istrain:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return loss.detach().cpu().item(), (logits.argmax(1)[mask]==graph.y[mask]).detach().cpu().sum()


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
            accs = main_multi_graph(NN, MFUNC[mn], nlayer)
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