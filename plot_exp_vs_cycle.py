import matplotlib.pyplot as plt
import os, torch
from matplotlib import colormaps

import matplotlib
font = {'family' : 'Times New Roman',
        # 'weight' : 'bold',
        'size'   : 13}

matplotlib.rc('font', **font)

tag = 'expressiveness_vs_cycle_len_eq'

markers = ['X', '*', '^', 'd', 's', 'o', '1', 'P', 'v']
cyc_len_num = [83, 63, 10, 6, 1, 1]
orgxs = [3, 4, 5, 6, 7, 8]
xs = [x*4 for x in orgxs]
fns = []
sorti = []
mns = []
for fn in os.listdir('.'):
    if not fn.endswith('.log'): continue
    if tag not in fn: continue
    mn = fn.split('_')[-1][:-4]
    if 'h2' in mn:
        sorti.append(int(mn[-1]))
        mns.append(f'{mn[-1]}-DeN')
    else:
        mns.append('1-/2-WL')
        sorti.append(0)
    fns.append(fn)

colormaps()
mi = 0
fns = [fns[i] for i in torch.LongTensor(sorti).argsort()]
mns = [mns[i] for i in torch.LongTensor(sorti).argsort()]
print(mns)
fig, ax1 = plt.subplots(figsize=(5,4))
color = 'tab:red'
ax2 = ax1.twinx()
# ax2.bar(xs, cyc_len_num, facecolor='None', edgecolor=color, hatch="//")
ax2.plot(xs, cyc_len_num, marker='_', linestyle='None', markersize=14, color=color)
ax2.set_ylabel('cycle number', color=color)
ax2.tick_params(axis='y', labelcolor=color)
for fn,mn in zip(fns,mns):
    with open(fn, 'r') as f:
        ratio = f.read().split('\n')[-2]
    if 'Ratio: ' != ratio[:7]: continue
    ratio = ratio.replace('Ratio: ', '')[1:-1].split(', ')
    ratio = [float(r) for r in ratio]
    ax1.plot([x+(mi-3.5)*0.6 for x in xs], ratio, markersize=7, marker=markers[mi], label=mn, color=colormaps['gist_gray']((mi*2)/30))
    mi += 1

ax1.set_ylabel('test passed ratio')
ax1.set_xlabel('cycle length')
ax1.set_xticks(xs, labels=orgxs)


fig.legend()
fig.tight_layout()
plt.savefig('exp_vs_cycle.svg')
