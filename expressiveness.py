import networkx as nx
import random, torch, os
from torch_geometric.utils import to_networkx
from datasets import get_data
import itertools
from multiprocessing import Pool
from collections import Counter
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from expressiveness_config import DEN_K,method,K,MAX_ITER,USE_DEN,PARALLEL_DEN,kwl_jobn,mtag,DEN_HASH_LEVEL,ONLY_INIT
from kfactor import get_kfactor
from networkx.classes.graph import Graph

def main():
    # names = ['cexp', 'exp', 'graph8c', 'mutag', 'nci1', 'enzymes', 'ptc_fm', 'imdb-binary']#
    # names = ['ptc_fm', 'imdb-binary']
    names = ['enzymes']#
    # names = ['graph8c']
    limited_test = ['graph8c', 'dd', 'proteins']#]#], 'sr25' 'graph8c'
    sp = 0
    saver = f"temp/noniso_ratio"
    os.makedirs(saver, exist_ok=True)
    os.makedirs("temp/init_color", exist_ok=True)
    global tempdir, init_colorf
    
    # for name in names[:sp]:
    #     if name in limited_test: 
    #         tag = f"{method}-TestOne{mtag}_K{K}_maxiter{MAX_ITER}_{name}"
    #     else:
    #         tag = f"{method}{mtag}_K{K}_maxiter{MAX_ITER}_{name}"
    #     tempdir = f"temp/wrong_iso_test/{tag}"
    #     init_colorf = f"temp/init_color/{method}{mtag}_K{K}_{name}.zip"
    #     os.makedirs(tempdir, exist_ok=True)

    #     graphs_iso_test = compare_exp_graph(name)
    #     with open(f"{saver}/{tag}.txt", 'w') as f:
    #         f.write(f"{sum(graphs_iso_test)} / {len(graphs_iso_test)} = {sum(graphs_iso_test)/len(graphs_iso_test)}")
            
    for name in names[sp:]:
        if name in limited_test: 
            tag = f"{method}-TestOne{mtag}_K{K}_maxiter{MAX_ITER}_{name}"
        else:
            tag = f"{method}{mtag}_K{K}_maxiter{MAX_ITER}_{name}"

        tempdir = f"temp/wrong_iso_test/{tag}"
        init_colorf = f"temp/init_color/{method}{mtag}_K{K}_{name}.zip"
        os.makedirs(tempdir, exist_ok=True)
        
        data = get_data(name, 0)
        if hasattr(data[0], "y") and data.num_classes >= 2: 
            graphs_iso_test = compare_diff_class(data, name)
            print(f"{sum(graphs_iso_test)} / {len(graphs_iso_test)} = {sum(graphs_iso_test)/len(graphs_iso_test)}")
        # else:
        #     graphs_iso_test = compare_all_graph(data, name, name in limited_test)
        ## init colors
        # graphs_iso_test = compare_all_graph(data, name, name in limited_test)
        continue
        with open(f"{saver}/{tag}.txt", 'w') as f:
            f.write(f"{sum(graphs_iso_test)} / {len(graphs_iso_test)} = {sum(graphs_iso_test)/len(graphs_iso_test)}")
    exit()

def compare_all_graph(data, name, limited=False):
    # compare_parallel_n = 15
    t_results = []
    graphs_iso_test = []
    wrong_set = []
    correct_set = []
    if os.path.exists(init_colorf):
        init_color_list = torch.load(init_colorf)
    else:
        init_color_list = [[None, None] for _ in range(len(data))]
        
    print(datetime.now(), len([1 for c, d in init_color_list if c is None or d is None]), f"graphs has no color, total {len(init_color_list)} graphs")
    if limited: 
        gis = [0]
    else:
        gis = trange(len(data), desc="Test graphs with all pairs of graph")
    for gi in gis:
        graph1 = to_networkx(data[gi])
        wrong_num = 0
        gjs = list(range(len(data)))
        random.shuffle(gjs)
        if limited: gjs = tqdm(gjs, desc="Test one graph to all others")
        for gj in gjs:
            if gj == gi: continue
            if f"{gi}-{gj}" in correct_set: continue
            if f"{gi}-{gj}" in wrong_set: 
                wrong_num += 1
                t_res = True
                if not limited: break
            else:
                init_c1, init_d1 = init_color_list[gi]
                init_c2, init_d2 = init_color_list[gj]
                graph2 = to_networkx(data[gj])
                t_res, init_c1, init_c2, init_d1, init_d2 = compare_graphs(graph1, graph2, method, K, init_c1, init_c2, init_d1, init_d2)
                assert init_c1 is not None, f"{gi}"
                assert init_d1 is not None, f"{gi}"
                assert init_c2 is not None, f"{gj}"
                assert init_d2 is not None, f"{gj}"
                init_color_list[gi] = [init_c1, init_d1]
                init_color_list[gj] = [init_c2, init_d2]
            t_results.append(t_res)

            if t_results[-1]: 
                wrong_num += 1
                wrong_set += [f"{gi}-{gj}", f"{gj}-{gi}"]
                if not limited: break
            else:
                correct_set += [f"{gi}-{gj}", f"{gj}-{gi}"]
            if limited: graphs_iso_test.append(t_results[-1] is False)
        if not limited: graphs_iso_test.append(wrong_num==0)

    torch.save(init_color_list, init_colorf)
    return graphs_iso_test

def compare_diff_class(data, name):
    t_results = []
    groups = []
    graphs_iso_test = []
    wrong_set = []
    correct_set = []
    if os.path.exists(init_colorf):
        init_color_list = torch.load(init_colorf)
        print(datetime.now(), "Loaded pre-initialized colors for WL tests")
    else:
        init_color_list = [[None, None] for _ in range(len(data))]
        print(datetime.now(), "No pre-initialized colors for WL tests, will init during test")
    print(datetime.now(), len([1 for c, d in init_color_list if c is None or d is None]), f"graphs has no color, total {len(init_color_list)} graphs")
    global g1id, g2id
    group_dataid = []
    for y in range(data.num_classes):
        groups.append([])
        group_dataid.append([])
        for i in [j for j in range(len(data)) if data[j].y == y]:
            groups[-1].append(to_networkx(data[i]))
            group_dataid[-1].append(i)

    for gi in trange(len(groups), desc="Test graphs with different class"):
        for gii in trange(len(groups[gi])):
            graph1 = groups[gi][gii]
            g1id = group_dataid[gi][gii]
            wrong_num = 0
            for gj in range(len(groups)):
                if gj == gi: continue
                for gji, graph2 in enumerate(groups[gj]):
                    g2id = group_dataid[gj][gji]
                    if f"{g1id}-{g2id}" in correct_set: continue
                    if f"{g1id}-{g2id}" in wrong_set: 
                        wrong_num += 1
                        t_res = True
                        break
                    init_c1, init_d1 = init_color_list[g1id]
                    init_c2, init_d2 = init_color_list[g2id]
                    t_res, init_c1, init_c2, init_d1, init_d2 = compare_graphs(graph1, graph2, method, K, init_c1, init_c2, init_d1, init_d2)
                    init_color_list[g1id] = [init_c1, init_d1]
                    init_color_list[g2id] = [init_c2, init_d2]
                    t_results.append(t_res)
                    if t_results[-1]: 
                        wrong_num += 1
                        wrong_set += [f"{g1id}-{g2id}", f"{g2id}-{g1id}"]
                        break
                    else:
                        correct_set += [f"{g1id}-{g2id}", f"{g2id}-{g1id}"]
                if wrong_num != 0:
                    break
            graphs_iso_test.append(wrong_num==0)

    torch.save(init_color_list, init_colorf)
    return graphs_iso_test


def compare_exp_graph(name):
    data = get_data(name, 0)
    if os.path.exists(init_colorf):
        init_color_list = torch.load(init_colorf)
    else:
        init_color_list = [[None, None] for _ in range(len(data))]
    
    graphs_iso_test = []
    t_results = []
    for gi in trange(int(len(data)/2), desc="Test EXP graphs"):
        graph1 = to_networkx(data[gi*2])
        graph2 = to_networkx(data[gi*2+1])
        init_c1, init_d1 = init_color_list[gi*2]
        init_c2, init_d2 = init_color_list[gi*2+1]
        t_res, init_c1, init_c2, init_d1, init_d2 = compare_graphs(graph1, graph2, method, K, init_c1, init_c2, init_d1, init_d2)
        init_color_list[gi*2] = [init_c1, init_d1]
        init_color_list[gi*2+1] = [init_c2, init_d2]
        t_results.append(t_res)
        if t_results[-1]:
            graphs_iso_test.append(False)
        else:
            graphs_iso_test.append(True)
        
    torch.save(init_color_list, init_colorf)
    return graphs_iso_test


def compare_graphs(G1, G2, method, k, init_c1, init_c2, init_d1, init_d2, verbose=False):
    if not isinstance(G1, Graph): G1 = to_networkx(G1)
    if not isinstance(G2, Graph): G2 = to_networkx(G2)
    methods = {
        'WL': WL,
        'kWL': kWL
    }

    # If two graphs have different numbers of nodes they cannot be isomorphic
    # if len(G1.nodes()) != len(G2.nodes()):
    #     if verbose:
    #         print('Non-Isomorphic by different number of nodes!')
    #     return False, init_c1, init_c2, init_d1, init_d2
    
    if method == 'kWL': 
        global kWL_neighbor_cache
        kWL_neighbor_cache = {}
    print("G1 id", g1id)
    c1, init_c1 = methods[method](G1, k, init_c1, verbose)
    
    if method == 'kWL': 
        kWL_neighbor_cache = {}
    print("G2 id", g2id)
    c2, init_c2 = methods[method](G2, k, init_c2, verbose)
    if not USE_DEN:
        test_result = c1==c2
        d1 = d2 = init_d1 = init_d2 = ''
    else:
        d1, init_d1 = methods[method](G1, k, init_d1, verbose, use_den=True)
        d2, init_d2 = methods[method](G2, k, init_d2, verbose, use_den=True)
        test_result = (c1==c2) and (d1==d2)
    
    if test_result:
        try:
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            nx.draw(G1.to_undirected(), ax=ax[0])
            ax[0].set_title(str(g1id)+":"+str(c1)+str(d1)+f"\n, K factor={get_kfactor(G1)}")
            nx.draw(G2.to_undirected(), ax=ax[1])
            ax[1].set_title(str(g2id)+":"+str(c2)+str(d2)+f"\n, K factor={get_kfactor(G2)}")
            fig.suptitle(f"Isomorphic: {test_result}")
            fig.savefig(f'{tempdir}/graph_compare_{g1id}_{g2id}.png')
            fig.savefig(f'{tempdir}/graph_compare_{g1id}_{g2id}.svg')
            plt.close()
        except:
            plt.close()
    return test_result, init_c1, init_c2, init_d1, init_d2


def wl_oneiter(G, n, node, glabels, find_neighbors_func):
    label = str(glabels[node])
    s2 = []
    for neighbor in find_neighbors_func(G, n, node):
        s2.append(glabels[neighbor])
    s2.sort()
    for i in range(len(s2)):
        label += "_" + str(s2[i])
    return label

def base_WL(G_, verbose, n_set, initial_colors_func, find_neighbors_func):    
    max_iter = MAX_ITER # maximum iteration number
    if verbose:
        print('-----------------------------------')
        print('Starting the execution for the graph')
    G, n = n_set(G_)
    colors = initial_colors_func(G, n) # colors = [hashable item]
    if ONLY_INIT:
        return [0], colors
    if verbose:
        print(f'Initial Color hashes: \n {colors} \n')
    
    def remap_label(labels):
        new_labels = {}
        new_label = 1
        remaped = {}
        for k in labels:
            if labels[k] not in new_labels:
                new_labels[labels[k]] = new_label
                new_label += 1
            remaped[k] = new_labels[labels[k]]
        return remaped
            
    labels = {} # store unique color
    glabels = remap_label(colors) # hash the color
    glabelsCount = 1
    newlabel = 1 # ID for unique color 
    done = False
    iter = 0
    print("nodes", n)
    while (not (done) and iter < max_iter):
        iter += 1
        glabelsNew = {}
        glabelsCountNew = 0
        for node in n:
            label = wl_oneiter(G, n, node, glabels, find_neighbors_func)
            if not (label in labels):
                labels[label] = newlabel
                newlabel += 1
                glabelsCountNew += 1
            glabelsNew[node] = labels[label]
        if glabelsCount == glabelsCountNew:
            done = True
        else:
            glabelsCount = glabelsCountNew
            glabels = glabelsNew.copy()
        print(iter)
        print(torch.LongTensor(list(glabels.values())).bincount().tolist())
    g0labels = []
    for node in n:
        g0labels.append(glabels[node])
    g0labels.sort()
    # hashed color (label) values are non-senseful for representing a graph, so using sorted bincount of labels
    c = torch.LongTensor(g0labels).bincount().tolist()
    # OR: hashed color (label) values are senseful for representing a graph, so using sorted labels
    # c = torch.LongTensor(g0labels).tolist()
    c.sort()
    return c, colors


def get_de(G, ni, nj, k):
    return len(list(nx.all_simple_paths(G, source=ni, target=nj, cutoff=k))) - 1

def de_initialize(G: Graph, n):
    colors = {}
    inputs = []
    nodes = []
    outputs = []
    refer = []
    passed = {}
    for i in n:
        for j in G.neighbors(i):
            nodes.append(i)
            inputs.append((G, i, j, DEN_K))
            if f'{i}-{j}' not in passed or f'{j}-{i}' not in passed: 
                passed[f'{i}-{j}'] = len(refer)
                passed[f'{j}-{i}'] = len(refer)
                refer.append(-1)
            else:
                refer.append(passed[f'{i}-{j}'])

    if not PARALLEL_DEN:
        for (G, i, j, den_k), refer_id in zip(inputs, refer):
            if refer_id == -1: 
                outputs.append(get_de(G, i, j, den_k))
            else:
                outputs.append(outputs[refer_id])
    else:
        compute_input = [inp for inp, refer_id in zip(inputs, refer) if refer_id != -1]
        with Pool(30) as p:
            out = list(p.starmap(get_de, compute_input))
        outputs = [None for _ in range(len(refer))]
        i = 0
        for ri in range(len(refer)):
            if refer[ri] == -1:
                outputs[ri] = out[i]
                i += 1
            else:
                outputs[ri] = outputs[refer[ri]]
    ## Not HASH, sum up all de
    if DEN_HASH_LEVEL <= 1:
        for i, den in zip(nodes, outputs):
            if i not in colors: colors[i] = 0
            colors[i] += den
    ## HASH De list
    if DEN_HASH_LEVEL == 2:
        for i, den in zip(nodes, outputs):
            if i not in colors: colors[i] = ''
            colors[i] += f"-{den}" # HASH{De1, De2, ...}
    ## HASH with Degree (1-De)
    if DEN_HASH_LEVEL > 0:
        for i in colors:
            colors[i] = f"{G.degree(i)}-{colors[i]}"
    return colors

def wl_find_neighbors(G, n, node):
    return G.neighbors(node)
    
def WL(G, k=2, init_color=None, verbose=False, use_den=False):
    def n_set(G):
        G = nx.convert_node_labels_to_integers(G)
        return G, list(G.nodes())
    
    def set_initial_colors(G, n):
        if init_color is not None: 
            return init_color
        if not use_den:
            return {i: G.degree[i] for i in n}
        else:
            colors = de_initialize(G, n)
            return colors#, {"i-j-...": o for (_, i, j, _), o in zip(inputs, outputs)}

    return base_WL(G, verbose, n_set, set_initial_colors, wl_find_neighbors)

def kwl_find_neighbors(G, V_k, node):
    if node in kWL_neighbor_cache: 
        return kWL_neighbor_cache[node]
    else:
        return [n for n in V_k if len(set(n) - set(node)) == 1]

def intlist_to_str(intlist):
    strlist = [str(intlist) for i in intlist]
    return "".join(strlist)

def kWL(G, k, init_color=None, verbose=False):
    def n_set(G):
        G = nx.convert_node_labels_to_integers(G)
        V = list(G.nodes())
        V_k = [comb for comb in itertools.combinations(V, k)]
        # print(datetime.now(), f"initialize {K}-WL neighbors")
        with Pool(kwl_jobn) as p:
            outputs = list(p.starmap(kwl_find_neighbors, [(None, V_k, comb) for comb in V_k]))
        for out, comb in zip(outputs, V_k):
            kWL_neighbor_cache[comb] = out
        # print(datetime.now(), f"initialize {K}-WL neighbors, done")
        # for comb in tqdm(V_k, desc="initialize {K}-WL neighbors"):
        #     kWL_neighbor_cache[comb] = kwl_find_neighbors(None, V_k, comb)
        return G, V_k

    def set_initial_colors(G, V_k):
        if init_color is not None: return init_color
        # print(datetime.now(), f"initialize {K}-WL colors")
        with Pool(kwl_jobn) as p:
            outputs = list(p.starmap(WL, [(get_subgraph(G, comb),) for comb in V_k]))
        # print(datetime.now(), f"initialize {K}-WL colors, done")
        return {V_k[i]: intlist_to_str(outputs[i]) for i in range(len(V_k))}
        # return {comb: intlist_to_str(WL(G.subgraph(nodes=comb))) for comb in V_k}

    return base_WL(G, verbose, n_set, set_initial_colors, kwl_find_neighbors)

def get_subgraph(G, largest_wcc):
    SG = G.__class__()
    SG.add_nodes_from((n, G.nodes[n]) for n in largest_wcc)
    if SG.is_multigraph():
        SG.add_edges_from((n, nbr, key, d)
            for n, nbrs in G.adj.items() if n in largest_wcc
            for nbr, keydict in nbrs.items() if nbr in largest_wcc
            for key, d in keydict.items())
    else:
        SG.add_edges_from((n, nbr, d)
            for n, nbrs in G.adj.items() if n in largest_wcc
            for nbr, d in nbrs.items() if nbr in largest_wcc)
    SG.graph.update(G.graph)
    return SG

if __name__ == "__main__":
    main()
