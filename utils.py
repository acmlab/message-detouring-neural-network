import torch
import networkx as nx
import numpy as np


def get_all_cycle(adj_mat):
    G = nx.from_numpy_array(adj_mat)
    all_cycle = []
    for ni in range(adj_mat.shape[0]):
        for nj in np.where(adj_mat[ni]>0)[0]:
            for path in nx.all_simple_paths(G, source=ni, target=nj):
                if len(path) > 2:
                    all_cycle.append(path)
    return all_cycle
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # G = nx.grid_2d_graph(10, 10)
    G = nx.fast_gnp_random_graph(20, 0.15)
    # G = nx.circulant_graph(20, [1])


    adj = nx.to_numpy_array(G)
    adj = torch.from_numpy(adj)

    ni = 10
    # opos = nx.spring_layout(G, iterations=100, seed=39775)
    opos = nx.circular_layout(G)
    pos = np.stack(list(opos.values()))

    fig, axes = plt.subplots(2,5,figsize=(30,6))
    axes = axes.reshape(-1)
    nx.draw(G, pos=opos, ax=axes[0])
    sp = 1

    for nj in np.where(nx.to_numpy_array(G)[ni]>0)[0]:
        print(nj)
        for path in nx.all_simple_paths(G, source=ni, target=nj):
            if len(path) > 2:
                # plt.plot(pos[path[1:], 0], pos[path[1:], 1], 'o', color='red')
                print(path)
                if sp < 10:
                    # nx.draw(nx.subgraph(G, path), pos={p:opos[p] for p in path}, ax=axes[sp])\
                    nx.draw(G, nodelist=path, edgelist=[(path[pi], path[pi+1]) for pi in range(len(path)-1)]+[(path[-1], path[0])], ax=axes[sp], pos=opos)
                    axes[sp].plot(pos[ni, 0], pos[ni, 1], '*', color='yellow')
                sp += 1
            else:
                axes[0].plot(pos[path[1:], 0], pos[path[1:], 1], 'o', color='red')
                axes[0].plot(pos[ni, 0], pos[ni, 1], '*', color='yellow')
            # print(path)
            
        
            
    plt.savefig(f'temp_graphplot.png')
    plt.close()
    exit()
