
def khop_neighbor(adj_mat, k, ni):
    '''
    k=0 is the direct neighborhood
    k=1 is the 1-hop neighborhood (shortest path length=2)
    k-hop neighborhood has the shortest path length=k+1
    '''
    # passed = torch.ones_like(adj_mat)
    # passed.fill_diagonal_(0) # node itself has been walked in the initialization
    a = adj_mat 
    a.fill_diagonal_(0)
    # passed[torch.where(a>0)] = 0 # record passed nodes
    # D = torch.zeros_like(adj_mat)
    # D.fill_diagonal_(adj_mat.sum(1))
    D = torch.diag(adj_mat.sum(1))
    detour_dis = torch.zeros_like(adj_mat)
    detour_dis[torch.where(a>0)] = 1
    prea = D
    # path_candidates = [torch.where(a[ni]>0)[0].tolist()]
    cycle_candidates = []
    cycle_candidate_l = []
    for ki in range(k):
        oa = a # walked adj
        a = a @ adj_mat # walked @ original adj = walk once
    
        # if _ < k-1: prea = a
        a = a - prea # remove occupied paths, multiple walks can still through walked paths
        # khop_detour = detour_dis[torch.where(a>0)]
        # khop_detour.clip_(min=k+1)
        # detour_dis[torch.where(a>0)] = khop_detour
        # a = a * passed # remove passed nodes
        # passed[torch.where(a>0)] = 0 # record passed nodes
        # path_candidates.append(torch.where(a[ni]>0)[0].tolist()) # all possible khop nodes
        if ki > 0:
            f1 = a[ni]>0
            f2 = adj_mat[ni]>0
            cycle_candidates.append(torch.where(f1&f2)[0])
            cycle_candidate_l.append(ki)
        
        prea = oa + torch.diag(oa.sum(1))
    
    
    # def build_path(candidates, all_path):
    #     if len(all_path) == 0: 
    #         all_path = [[node] for node in candidates]
    #     else:
    #         prevn = [-1 for _ in all_path]
    #         for pi in range(len(all_path)):
    #             pn = prevn[pi]
    #             for node in candidates:
    #                 if adj_mat[all_path[pi][pn], node] > 0:
    #                     d = D[all_path[pi][pn], all_path[pi][pn]] 
    #                     if d <= 1: 
    #                         all_path.append(all_path[pi]+[node])
    #                     else:
    #                         all_path[pi].append(node)
    #                         D[all_path[pi][pn], all_path[pi][pn]] = d - 1
    #                         prevn[pi] -= 1

    #     return all_path
        
    # all_path = []
    # for path_candidate in path_candidates:
    #     all_path = build_path(path_candidate, all_path)
        
    # return torch.where(a>0), all_path
    return a, cycle_candidates, cycle_candidate_l
    # return a
    # detour_dis.fill_diagonal_(0)
    # return detour_dis
# detour_dis = khop_neighbor(adj, 3)

for k in range(20):
# for k in detour_dis.unique():
    if k==0: continue
    plt.figure()
    nx.draw(G, pos=opos)
    A, cycle_candidates, cycle_candidate_l = khop_neighbor(adj, k, ni)
    khop_id = torch.where(A>0)
    An = A[ni, A[ni]>0]
    # khop_id = torch.where(detour_dis==(k))
    ni_khop = khop_id[1][khop_id[0]==ni]
    pos_khop = pos[ni_khop]
    plt.plot(pos[ni, 0], pos[ni, 1], '*', color='yellow')
    
    if len(pos_khop.shape) == 1: pos_khop = pos_khop[None, :]
    
    for an, (x, y) in enumerate(pos_khop):
        print(k+2 in [len(l) for l in nx.all_simple_paths(G, source=ni, target=ni_khop[an].item())])
        plt.text(x, y, f"{k}")
        plt.text(x, y+0.1, f"{An[an]}")
    # for pi, path in enumerate(paths):
    #     for x, y in pos[path]:
    #         plt.text(x, y+0.1, f"path-{pi}")
    for cc, cl in zip(cycle_candidates, cycle_candidate_l):
        for pt in cc:
            x = (pos[ni, 0] + pos[pt, 0])/2
            y = (pos[ni, 1] + pos[pt, 1])/2
            plt.text(x, y, f'cycle_l={cl}')
    plt.savefig(f'temp_graphplot_{k}.png')
    plt.close()

