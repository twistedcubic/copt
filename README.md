
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

There is a [data](data) directory used by the scripts to write data to. There is some generated sample data provided.

## Dependencies

PyTorch 1.1+
numpy
networkx

Depending on the functionalities one wishes to run, additional dependencies include:
[Gromov Wasserstein by Vayer et al](https://github.com/tvayer/FGW), can be placed as "gromov" in directory above this one.

