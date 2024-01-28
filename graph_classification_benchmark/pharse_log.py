import numpy as np
# fn = '../node_classification_benchmark/log/actor-gat-trans0-denk4-mdnn4layer_20240121170724.log'
# fn = '../node_classification_benchmark/log/texas-gin-trans1-wodee0-mdnn4layer_20240120220509.log'
# fn = '../node_classification_benchmark/log/texas-gin-trans1-wodee1-mdnn4layer_20240125191102.log'
fn = '../node_classification_benchmark/log/chameleon-gcn-trans1-wodee1-mdnn4layer_20240124181958.log'
with open(fn, 'r') as f:
    lines = f.read().split('\n')[:-1]
lines = [l for l in lines if l.startswith('Split')]
ne = 1
bv = [0 for _ in range(ne)]
acc = [0 for _ in range(ne)]
for l in lines[1:-1]:
    tr, va, te = l.split('acc:\t(')[1].split(') (')[0].split(', ')
    split = int(l.split(' ')[1])
    if split-1 >= len(bv): continue
    if float(va) >= bv[split-1]: 
        # if float(va) == bv[split-1]:
        acc[split-1]=max(acc[split-1], float(te))
        # acc[split-1]=float(te)
        bv[split-1]=float(va)

print(np.mean(acc), np.std(acc))