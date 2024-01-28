## Message detouring

### Data

Download data splits for graph classification from this [github repo](https://github.com/diningphil/gnn-comparison/tree/master/data_splits), and put `*_splits.json` to `graph_classification_benchmark/data`. Data will be automatically downloaded when first call `main.py`.

### Expressive power

Related codes:
`expressiveness.py`, `plot_exp.py`

### Graph kernel

Related codes:
`graph_kernel_benchmark/detour_number_wl.py`

```
cd graph_kernel_benchmark
python detour_number_wl.py 
```

### Graph classification

Related codes:
`graph_classification_benchmark/main.py`

```
cd graph_classification_benchmark
python main.py [data_name] [device] [1 or 0: skip connection or not] [baseline] [1 or 0: mpnn or mdnn] [1 or 0: use transformer or not] [1 or 0: w/o DeE or not] [k, default=2]
```

### Node classification

Related codes:
`node_classification_benchmark/main.py`

```
cd node_classification_benchmark
python main.py [data_name] [device] [1 or 0: skip connection or not] [baseline] [1 or 0: mpnn or mdnn] [1 or 0: use transformer or not] [1 or 0: w/o DeE or not] [k, default=2]
```