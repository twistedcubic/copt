
"""
Demo for using COPT for graph sketching.
"""

import torch
import numpy as np
import utils
import runGraph
import graph

def sketch_graph(args):
    data_dim = 20
    lo_dim = 5
        
    g1 = utils.create_graph(data_dim, 'random_regular')
    
    #args.n_epochs = 300 <--parameters like this can be set here or in command line
    args.Lx = utils.graph_to_lap(g1)
    args.m = len(args.Lx)
    args.n = lo_dim
    # sketch graphs of lo_dim.
    # Returns optimization loss, transport plan P, and Laplacian of sketched graph
    loss, P, Ly = graph.graph_dist(args, plot=False)
    print('sketched graph Laplacian {}'.format(Ly))
    #can convert Ly to a networkx graph with utils.lap_to_graph(Ly)
    return loss, P, Ly

if __name__ == '__main__':
    args = utils.parse_args()
    if args.fix_seed:
        torch.manual_seed(0)
        np.random.seed(0)
    sketch_graph(args)
    
    
