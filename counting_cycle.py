import torch

# dn = 'mutag'

def count_cycle(dn):
    cl_lists = torch.load(f'{dn}_cycle_length_list.zip')
    tags = {}
    label = 1
    labels = []
    for gi, cl_list in enumerate(cl_lists):
        if len(cl_list) < 3 or sum(cl_list) == 0: 
            tag = 'noncyc'
        else:
                tag = ''.join([str(l) for l in set(torch.where(cl_list>0)[0].tolist())])
        if tag not in tags:
            if dn not in ['texas', 'wisconsin', 'cornell', 'actor', 'squirrel', 'chameleon']:
                tags[tag] = label
                label += 1 
            else:
                 tags[tag] = cl_list[torch.where(cl_list>0)[0]].tolist()
        
        labels.append(tags[tag])
    print(dn)
    print(tags)
    if dn not in ['texas', 'wisconsin', 'cornell', 'actor', 'squirrel', 'chameleon']:
        print("bincount", torch.LongTensor(labels).bincount().tolist()[1:])
    
for dn in ['texas', 'wisconsin', 'cornell', 'actor']:#, 'squirrel', 'chameleon' , 'exp', 'cexp', 'mutag', 'nci1', 'enzymes', 'ptc_fm', 'imdb-binary', 'dd']:
# for dn in ['squirrel']:
    count_cycle(dn)