import torch, os
from datasets import get_data
from torch_geometric.utils import to_networkx
from expressiveness import compare_graphs
from tqdm import tqdm, trange
from expressiveness_config import DEN_K,method,K,MAX_ITER,USE_DEN,PARALLEL_DEN,kwl_jobn,mtag,DEN_HASH_LEVEL,ONLY_INIT
import matplotlib.pyplot as plt
from kfactor import get_kfactor
dn = 'graph8c'

def cycle_vs_cycle(tgt_range1, tgt_range2):
    group_list1 = group_data(tgt_range1)[1:]
    group_list2 = group_data(tgt_range2)[1:]
    
    # data = get_data(dn, 0)
    # run_groups = []
    # tgt_range = ["Non-cycle"] + tgt_range
    for i in trange(len(group_list1)):
        length, ratio = compare_groups(group_list1[i], group_list2[i])
        print(f"Cyc len == [Noncycle]+{tgt_range1[i]} vs Cyc len == [Noncycle]+{tgt_range2[i]}: Pair num: {length}, pass ratio: {ratio}, not passed num: {round(length*(1-ratio))}")
        print()

def all_vs_cycle(tgt_range):
    group_list = group_data(tgt_range)[1:]
    data = get_data(dn, 0)
    run_groups = []
    tgt_range = ["Non-cycle"] + tgt_range
    for i in trange(len(group_list)):
        length, ratio = compare_groups(list(range(len(data))), group_list[i])
        print(f"Cyc len == {tgt_range[i]} vs All others: Pair num: {length}, pass ratio: {ratio}, not passed num: {round(length*(1-ratio))}")
        print()
        # for j in trange(len(group_list)):
        #     if f"{i}-{j}" in run_groups or f"{j}-{i}" in run_groups: continue
        #     length, ratio = compare_groups(group_list[i], group_list[j])
        #     print(f"Cyc len == {tgt_range[i]} vs Cyc len == {tgt_range[j]}: Pair num: {length}, pass ratio: {ratio}, not passed num: {round(length*(1-ratio))}")
        #     run_groups.append(f"{i}-{j}")
        #     run_groups.append(f"{j}-{i}")

def group_data(tgt_range):
    cl_lists = torch.load(f'{dn}_cycle_length_list.zip')
    out = []
    for tgt_cl in tgt_range:
        gis = []
        nongis = []
        for gi, cl_list in enumerate(cl_lists):
            tgt = []
            if len(cl_list) < 3: 
                nongis.append(gi)
                continue
            # print(cl_list)
            for cl, cycle_num in enumerate(cl_list[3:]):
                cl = cl + 3
                if not isinstance(tgt_cl, list):
                    cond = cl == tgt_cl and cycle_num > 0
                else:
                    # cond = cl >= tgt_cl and cl <= tgt_cl_max
                    cond = cl in tgt_cl and cycle_num > 0
                # if not cond:
                #     print(cl, cycle_num)
                tgt.append(cond)
            if all(tgt): 
                gis.append(gi)
        if len(out) == 0:
            out += [nongis, nongis+gis]
        else:
            out.append(nongis+gis)
    # assert sum([len(o) for o in out]) == len(cl_lists), f"{sum([len(o) for o in out])} != {len(cl_lists)}, {[len(o) for o in out]}"
    return out
            
def compare_groups(group1, group2):
    data = get_data(dn, 0)
    init_colorf = f"temp/init_color/{method}{mtag}_K{K}_{dn}.zip"
    if os.path.exists(init_colorf):
        init_color_list = torch.load(init_colorf)
    data = [to_networkx(d) for d in data]
    # t_results = []
    wrong_set = []
    correct_set = []
    for gi in tqdm(group1):
        graph1 = data[gi]
        if get_kfactor(graph1) > 2: continue
        for gj in group2:
            if gj == gi: continue
            if f"{gi}-{gj}" in correct_set: continue
            if f"{gi}-{gj}" in wrong_set: continue
            graph2 = data[gj]
            if get_kfactor(graph2) > 2: continue
            init_c1, init_d1 = init_color_list[gi]
            init_c2, init_d2 = init_color_list[gj]
            assert init_c1 is not None and init_c2 is not None and init_d1 is not None and init_d2 is not None, f"{init_c1}, {init_c2}, {init_d1}, {init_d2}"
            t_res, init_c1, init_c2, init_d1, init_d2 = compare_graphs(graph1, graph2, method, K, init_c1, init_c2, init_d1, init_d2)
            # t_results.append(not t_res)
            if t_res: 
                wrong_set += [f"{gi}-{gj}", f"{gj}-{gi}"]
            else:
                correct_set += [f"{gi}-{gj}", f"{gj}-{gi}"]
    pair_len = len(correct_set) + len(wrong_set)
    pair_len = int(pair_len/2)
    return pair_len, (len(correct_set)/2)/pair_len

def main(tgt_cl = 3):
    cl_lists = torch.load(f'{dn}_cycle_length_list.zip')
    data = get_data(dn, 0)
    # graphs = []
    gis = []
    nongis = []
    if isinstance(tgt_cl, list):
        tgt_cl, tgt_cl_max = tgt_cl
    else:
        tgt_cl_max = None
    for gi, cl_list in enumerate(cl_lists):
        tgt = []
        if len(cl_list) == 0: 
            nongis.append(gi)
            continue
        # print(cl_list)
        for cl, cycle_num in enumerate(cl_list):
            if cycle_num == 0: continue
            if tgt_cl_max is None:
                cond = cl == tgt_cl
            else:
                cond = cl >= tgt_cl and cl <= tgt_cl_max
            if cond:
                tgt.append(True)
            else:
                tgt.append(False)
        if all(tgt) and len(tgt)>0: 
            # graphs.append(to_networkx(data[gi]))
            gis.append(gi)
    # print(len(nongis), len(cl_lists))
    # exit()
            
    # gis = nongis
    init_colorf = f"temp/init_color/{method}{mtag}_K{K}_{dn}.zip"
    if os.path.exists(init_colorf):
        init_color_list = torch.load(init_colorf)
    data = [to_networkx(d) for d in data]
    graphs_iso_test = []
    t_results = []
    wrong_set = []
    correct_set = []
    # for gi in trange(len(data)):
    for gi in tqdm(gis):
        graph1 = data[gi]
        wrong_num = 0
        for gj in nongis:
            if gj == gi: continue
            if f"{gi}-{gj}" in correct_set: continue
            if f"{gi}-{gj}" in wrong_set: 
                wrong_num += 1
                t_res = True
                t_results.append(not t_res)
                break
            else:
                init_c1, init_d1 = init_color_list[gi]
                init_c2, init_d2 = init_color_list[gj]
                assert init_c1 is not None and init_c2 is not None and init_d1 is not None and init_d2 is not None, f"{init_c1}, {init_c2}, {init_d1}, {init_d2}"
                graph2 = data[gj]
                t_res, init_c1, init_c2, init_d1, init_d2 = compare_graphs(graph1, graph2, method, K, init_c1, init_c2, init_d1, init_d2)
                t_results.append(not t_res)

            if t_res: 
                wrong_num += 1
                wrong_set += [f"{gi}-{gj}", f"{gj}-{gi}"]
                break
            else:
                correct_set += [f"{gi}-{gj}", f"{gj}-{gi}"]

        graphs_iso_test.append(wrong_num==0)
    try:
        # return len(gis), sum(graphs_iso_test)/len(graphs_iso_test)
        return len(t_results), sum(t_results)/len(t_results)
    except:
        return len(gis), 0

lengs, ratios = [], []
# tgt_range = list(range(3, 9))
tgt_range = [3, 4, 5, [6,7,8]]
# all_vs_cycle(tgt_range)
# cyclemix = [[i for i in range(3,6) if i != j]+[6,7,8] for j in range(3,6)]+[[3,4,5]]
# print(cyclemix)
cyclemix1 = [[i for i in range(3, j+1)] for j in range(3,6)]
cyclemix2 = [[i for i in range(j+1, 8)] for j in range(3,6)]
cycle_vs_cycle(cyclemix1+cyclemix2, cyclemix1+cyclemix2)
# cycle_vs_cycle([[3,4], [4,5,6,7,8], [3,5,6,7,8], [3,4,5,6,7,8]])
# cycle_vs_cycle([[5,6,7,8], [3,4], [4,5,6,7,8], [3,5,6,7,8], [3,4,5,6,7,8]])
# all_vs_cycle(cyclemix)
# for tgt in tqdm(tgt_range):
#     leng, ratio = main(tgt)
#     lengs.append(leng)
#     ratios.append(ratio)
#     print("Tgt cycle length ==", tgt, f"Passed Num: {torch.round(leng*(1-ratio)).item()}", "Ratio:", ratio)
# print("Cycle len", tgt_range)
# print("Length:", lengs)
# print("Ratio:", ratios)