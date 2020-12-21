
# COPT: Coordinated Optimal Transport for Graph Sketching

COPT is a novel distance metric between graphs defined via an optimization routine, computing a coordinated pair of optimal transport maps simultaneously. This is an unsupervised way to learn general-purpose graph representations, it can be used for both graph sketching and graph comparison.

For a sample run script, please see demo.py. For instance, to sketch a sample graph with 400 training steps and with fixed seed we can run:

`python demo.py --seed --n_epochs 400`

There are many other options to allow easy custom tuning. To see all command line options, see `utils.py`[utils.py] or run:

`python demo.py --h`

For instance, one can run COPT with:

`python searchGraph.py --hike --hike_interval 15`
`python searchGraph.py --hike --hike_interval 15 --grid_search --seed --compress_fac 4`

[`graph.py`](graph.py) contains core COPT routines for applications such as graph sketching and comparison.
[`runGraph.py`](runGraph.py), [`searchGraph.py`](searchGraph.py), etc contain various applications for COPT.

There is a [data](data) directory used by the scripts to write data to. There is some generated [sample data provided](data). Furthermore, if one wishes to generate graph data for other [named datasets](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets), one can run the `generateData.py` script with the dataset name such as:;
```
python generateData.py --dataset_type real --dataset_name BZR
```
A corresponding `lap.pt` data file will be created.

## Dependencies

PyTorch 1.1+
numpy
networkx
netlsd
grakel

To install PyTorch, please follow these [simple OS-specific instructions](https://pytorch.org/get-started/locally/).

The other packages can be installed via `pip`, e.g. `python -m pip install numpy networkx grakel netlsd`. Or by running
```
pip install -r requirements.txt
```

Depending on the functionalities one wishes to run, additional dependencies include:
[Gromov Wasserstein by Vayer et al](https://github.com/tvayer/FGW), can be placed as "gromov" in directory above this one.

