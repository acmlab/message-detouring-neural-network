import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import os
# plt.rcParams['font.family'] = 'Times New Roman'
r = 'temp/noniso_ratio'
import matplotlib
font = {'family' : 'Times New Roman',
        # 'weight' : 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)

mat = []
coln = []
rown = []
for rn in os.listdir(r):
    with open(os.path.join(r,rn), 'r') as f:
        res = float(f.read().split('\n')[0].split(' = ')[-1])
    # with open(os.path.join('temp/express_time',rn), 'r') as f:
    #     res = float(f.read().split('\n')[0].split(' = ')[-1])
    mn = rn.split('_')[0]
    itern = int(rn.split('_')[2].replace('maxiter',''))
    if 'WL-DeN' == mn: continue
    if 'H1DeN' in mn: continue
    if 'TestOne' in mn and 'graph8c' not in mn: continue
    if 'TestOne' in mn: 
        mn = mn.replace('-TestOne', '')
    if 'DeN' in mn: 
        itern = int(mn[-1])
        # mn = 'WL-DeN'
        mn = mn[:-1]
    elif 'kWL' in mn:
        mn = '3-'+mn[1:]
        itern = 1
    mn = mn+"$_{"+str(itern)+"}$"
    datan = rn.split('_')[-1].replace('.txt','')
    if mn not in rown: 
        rown.append(mn)
        mat.append([])
        rowi = len(rown) - 1
    else:
        rowi = rown.index(mn)
    
    if datan not in coln: 
        coln.append(datan)
        coli = len(coln) - 1
    else:
        coli = coln.index(datan)

    if coli+1 > len(mat[rowi]):
        mat[rowi] += [-0.01 for _ in range(coli+1-len(mat[rowi]))]
    mat[rowi][coli] = res
# print(mat)
for i in range(len(mat)):
    if len(coln) > len(mat[i]):
        mat[i] += [-0.01 for _ in range(len(coln)-len(mat[i]))]
mat = np.array(mat)

coln_remap = {'ptc_fm':'fm'}
sort_coln = ['exp','cexp', 'graph8c', 'sr25','mutag', 'nci1', 'enzymes', 'ptc_fm', 'imdb-binary']
new_coln, colid = [], []
for n in sort_coln:
    new_coln.append(n.upper())
    if n in coln_remap: n = coln_remap[n]
    colid.append(coln.index(n))

mn_passed = []
rowid = []
for ni, n in enumerate(rown):
    if ni in rowid: continue
    # if mn in mn_passed: continue
    mn = n.split('$_')[0]
    itern = int(n.split('_{')[-1][:-2])
    mnid = [ni]
    mnit = [itern]
    for nni, nn in enumerate(rown):
        if ni == nni: continue
        # if mn in mn_passed: continue
        if mn != nn.split('$_')[0]: continue
        mnid.append(nni)
        mnit.append(int(nn.split('_{')[-1][:-2]))
    
    rowid += [mnid[i] for i in np.argsort(mnit)]
    # mn_passed.append(mn)
new_rown = []
# new_rown = [rown[i] if 'WL' == rown[i].split('$_')[0] else rown[i].split('$_')[0] ]
for i in rowid:
    if 'WL' == rown[i].split('$_')[0]:
        new_rown.append("1-/2-"+rown[i])
    elif 'DeN' in rown[i].split('$_')[0]:
        new_rown.append(rown[i].split('_{')[-1][:-2]+"-"+rown[i].split('$_')[0].split('-')[1])
    elif '3-WL' == rown[i].split('$_')[0]:
        new_rown.append('3-WL')
    else:
        new_rown.append(rown[i])

# new_rown = [rown[i] for i in rowid]
mat = mat[rowid, :][:, colid]

remain_row = [7,8,9,10,11,12,13,0,1,2,3,4,5,6,-1]
remain_col = [0,1,4,5,6,7,8]
mat = mat[remain_row, :][:, remain_col]
new_rown = [new_rown[i] for i in remain_row]
new_coln = [new_coln[i] for i in remain_col]

##time
# 10:50, 42:40, 3:42


times = {
    '1-/2-WL$_{1}$': '?h?m',
    '1-/2-WL$_{2}$': '?h?m',
    '1-/2-WL$_{3}$': '?h?m',
    '1-/2-WL$_{4}$': '?h?m',
    '1-/2-WL$_{6}$': '?h?m',
    '1-/2-WL$_{8}$': '?min',
    '1-/2-WL$_{10}$': '?min',
    '2-DeN': '?h?m',
    '3-DeN': '?min',
    '4-DeN': '?min',
    '5-DeN': '?min',
    '6-DeN': '?min',
    '7-DeN': '?min',
    '8-DeN': '?min',
    '3-WL': '?d',
}

fig, ax1 = plt.subplots(figsize=(7,6))

sn.heatmap(mat, fmt='.2f', annot=True, xticklabels=new_coln, yticklabels=new_rown, cmap='coolwarm', ax=ax1)
# ax2 = ax1.twinx()
ax1.set_xticklabels(new_coln, rotation=45) 
plt.yticks(rotation=0) 
# ax2.set_ylim(ax1.get_ylim())
# ax2.set_yticks(ax1.get_yticks())
# ax2.set_yticklabels([times[mn] for mn in new_rown])

plt.tight_layout()
plt.savefig('expressiveness.svg')