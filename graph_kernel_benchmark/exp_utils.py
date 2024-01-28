import json, os, random
import numpy as np

from sklearn.svm import SVC
from sklearn.utils import Bunch, check_random_state
# from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

from grakel.utils import KMTransformer
from grakel.graph import is_adjacency as valid_matrix
from six.moves.collections_abc import Iterable

DEN_K = 2
DATALIST = [
    # 'MUTAG',
    # 'PTC_FM',
    # 'ENZYMES',
    'PROTEINS',
    # 'NCI1',
    # 'DD',
    # 'IMDB-BINARY',
]

random.seed(142857)

def cross_validate_Kfold_SVM(K, y,
                             n_iter=10, n_splits=10, C_grid=None, kfolder=None,
                             random_state=None, scoring="accuracy", fold_reduce=None):
    # Initialise C_grid
    if C_grid is None:
        C_grid = ((10. ** np.arange(-7, 7, 2)) / len(y)).tolist()
    elif type(C_grid) is np.array:
        C_grid = np.squeeze(C_grid)
        if len(C_grid.shape) != 1:
            raise ValueError('C_grid should either be None or a squeezable to 1 dimension np.array')
        else:
            C_grid = list(C_grid)

    # Initialise fold_reduce:
    if fold_reduce is None:
        fold_reduce = np.mean
    elif not isinstance(callable, fold_reduce):
        raise ValueError('fold_reduce should be a callable')

    # Initialise and check random state
    random_state = check_random_state(random_state)

    # Initialise sklearn pipeline objects
    if kfolder is None:
        kfolder = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        # Make all the requested folds
        nfolds = tuple(tuple(kfolder.split(y)) for _ in range(n_iter))
    else:
        nfolds = tuple(tuple(kfolder) for _ in range(n_iter))
    estimator = make_pipeline(KMTransformer(), SVC(kernel='precomputed'))


    out = list()
    for ks in K:
        mid = list()
        if valid_matrix(ks):
            pg = {"svc__C": C_grid, "kmtransformer__K": [Bunch(mat=ks)]}
        elif isinstance(ks, Iterable) and all(valid_matrix(k) for k in ks):
            pg = [{"svc__C": C_grid, "kmtransformer__K": [Bunch(mat=k)]} for k in ks]
        else:
            raise ValueError('Not a valid object for kernel matrix/ces')

        for kfolds in nfolds:
            fold_info = list()
            for train, test in kfolds:
                gs = GridSearchCV(estimator, param_grid=pg, scoring=scoring, n_jobs=None,
                                  cv=ShuffleSplit(n_splits=1,
                                                  test_size=0.1,
                                                  random_state=random_state)).fit(train, y[train])
                fold_info.append(gs.score(test, y[test]))
            mid.append(fold_reduce(fold_info))
        out.append(mid)
    return out

def get_splits(data, name=None, splitn=10, random_split=False):
    jpath = f"/ram/USERS/ziquanw/data/{name.lower()}/{name}_splits.json"
    trains = []
    vals = []
    tests = []
    dlen = len(data)
    if os.path.exists(jpath) and not random_split:
        splits = json.load(open(jpath, 'r'))
        assert splitn == len(splits), f"unsupport split file {jpath}"
        for split in splits:
            trains.append(
                split['model_selection'][0]['train']
            )
            vals.append(
                split['model_selection'][0]['validation']
            )
            tests.append(
                split['test']
            )
    else:
        ids = list(np.arange(dlen))
        random.shuffle(ids)
        splitlen = int(dlen/splitn)
        for i in range(splitn):
            if i < splitn-1:
                msid = ids[:i*splitlen] + ids[(i+1)*splitlen:] 
                testid = ids[i*splitlen:(i+1)*splitlen] 
            else:
                msid = ids[:i*splitlen]
                testid = ids[i*splitlen:]
            trainid = msid[len(testid):]
            valid = msid[:len(testid)]
            trains.append(trainid)
            vals.append(valid)
            tests.append(testid)
    # out = []
    # for a in arrays:
    #     out.append(a[trainid])
    # return np.array(trains), np.array(vals), np.array(tests)
    return trains, vals, tests