'''
Generate and skecth graphs used for downstream tasks.
Also parse and generate graph data from various real datasets, e.g. MSRC.
'''
import _init_paths
import utils
import graph
import lib.graph as gwGraph
import networkx as nx
import torch
from lib.ot_distances import Fused_Gromov_Wasserstein_distance
import numpy as np
import pickle
from tqdm import tqdm
import stochastic as st
import runGraph
import os
import grakel

import pdb

torch.set_default_tensor_type('torch.DoubleTensor')

def sketch_graph(graphs, dataset_cls, lo_dim, args):
    '''
    Run graph sketching.
    Input: graphs: graphs to be dimension-reduced for.
    '''
    args.n = lo_dim
    lo_graphs = []
    lo_cls = []
    args.n_epochs = 230 #250
    for i, g in enumerate(tqdm(graphs, desc='sketching')):
        args.Lx = utils.graph_to_lap(g) #graph.graph_dist(args, plot=False)        
        args.m = len(args.Lx)
        try:
            #rarely, 0.2% of time pytorch's eigenvalue finding doesn't converge
            loss, P, Ly = graph.graph_dist(args, plot=False)
        
        except RuntimeError as e:
            #    pdb.set_trace()
            print(e)
            continue
        lo_graphs.append(utils.lap_to_graph(Ly))
        lo_cls.append(dataset_cls[i])
    return lo_graphs, lo_cls

def generate_real_data(dataset_name='msrc'):
    '''
    Parse graphs and generate Laplacians from real datasets.
    '''
    
    dataset = grakel.datasets.fetch_dataset(dataset_name)
    graphs = dataset.data
    target = [] #dataset.target
    node_labels = []
    lap_l = []
    total_nodes = 0
    for j, graph in enumerate(graphs):
        g = grakel.Graph(graph_format='adjacency')
        #geaph[1] eg {1: 3, 2: 3, 3: 3
        #g_adj_l = g[0]        
        g.build_graph(*graph)
        lap = g.laplacian()
        #pdb.set_trace()
        g1=utils.lap_to_graph(lap.copy())
        if not nx.is_connected(g1):
            print('not connected!')
            continue
        node_label = np.zeros((len(lap))) #[-1]*len(g[1])
        if len(lap) < len(graph[1]):
            print('len except!')
            continue
        k_prev = -1
        for i, (k,l) in enumerate(graph[1].items()):
            assert k > k_prev
            k_prev = k
            #pdb.set_trace()
            try:
                #node_label[k-1] = l
                node_label[i] = l
            except Exception as e:
                print('e!', e)
                pdb.set_trace()
                break
            
        total_nodes += len(lap)
        node_labels.append(node_label)
        target.append(dataset.target[j])
        lap_l.append(torch.from_numpy(lap))

    print('dataset {} avg nodes {}'.format(dataset_name, total_nodes / len(lap_l)))
    pdb.set_trace()
    torch.save({'lap':lap_l, 'labels':node_labels, 'target':target}, '{}_lap.pt'.format(dataset_name)) #os.path.join('data', dataset_name)))
    return lap_l, node_labels, target

def test_FGW(args):
    """
    Fused Gromov-Wasserstein distance
    """
    args.m = 8
    args.n = 4
    if args.fix_seed:
        torch.manual_seed(0)
    #args.Lx = torch.randn(args.m*(args.m-1)//2)  #torch.FloatTensor([[1, -1], [-1, 2]])
    #args.Lx = realize_upper(args.Lx, args.m)
    #pdb.set_trace()    
    g = nx.stochastic_block_model([4,4],[[0.9,0.1],[0.1,0.9]], seed = 8576)
    #components = nx.connected_components(g)
    g.remove_nodes_from(list(nx.isolates(g)))
    args.m = len(g)
    Lx = nx.laplacian_matrix(g, range(args.m)).todense()
    args.Lx = torch.from_numpy(Lx).to(dtype=torch.float32) #+ torch.ones(args.m, args.m)/args.m
    args.n_epochs = 150
    '''
    g2 = nx.stochastic_block_model([4,4],[[0.9,0.1],[0.1,0.9]])    
    g2.remove_nodes_from(list(nx.isolates(g2)))
    args.n = len(g2)
    '''
    loss, P, L = graph.graph_dist(args, plot=False)
    if isinstance(L, torch.Tensor):
        L = L.numpy()
    np.fill_diagonal(L, 0)
    A = -L
    g2 = nx.from_numpy_array(A)
    
    gwdist = Fused_Gromov_Wasserstein_distance(alpha=0.8,features_metric='sqeuclidean')
    g = gwGraph.Graph(g)
    g2 = gwGraph.Graph(g2)    
    dist = gwdist.graph_d(g,g2)    
    print('GW dist ', dist)   

    ###
    g3 = nx.stochastic_block_model([4,4],[[0.9,0.1],[0.1,0.9]],seed=452)    
    g3.remove_nodes_from(list(nx.isolates(g3)))
    args.m = len(g3)
    Lx = nx.laplacian_matrix(g3, range(args.m)).todense()
    args.Lx = torch.from_numpy(Lx).to(dtype=torch.float32) #+ torch.ones(args.m, args.m)/args.m    
    loss2, P2, L2 = graph.graph_dist(args, plot=False)
    L=L2
    if isinstance(L, torch.Tensor):
        L = L.numpy()
    np.fill_diagonal(L, 0)
    A = -L
    g4 = nx.from_numpy_array(A)
    
    #gwdist = Fused_Gromov_Wasserstein_distance(alpha=0.8,features_metric='sqeuclidean')
    g3 = gwGraph.Graph(g3)
    g4 = gwGraph.Graph(g4)    
    dist = gwdist.graph_d(g3,g4)    
    print('GW dist ', dist)   
    
    pdb.set_trace()
    
def run_community_graph(args):

    args.m = 12
    args.n = 4
    if args.fix_seed:
        torch.manual_seed(0)
    #args.Lx = torch.randn(args.m*(args.m-1)//2)  #torch.FloatTensor([[1, -1], [-1, 2]])
    #args.Lx = realize_upper(args.Lx, args.m)
    #pdb.set_trace()
    g = nx.stochastic_block_model([6,6],[[0.9,0.1],[0.1,0.9]], seed = 8576)
    #components = nx.connected_components(g)
    g.remove_nodes_from(list(nx.isolates(g)))
    args.m = len(g)
    
    Lx = nx.laplacian_matrix(g, range(args.m)).todense()
    args.Lx = torch.from_numpy(Lx).to(dtype=torch.float32) #+ torch.ones(args.m, args.m)/args.m
    
    args.n_epochs = 370 #100
    graph.graph_dist(args)


def run_same_dim(args):
    """
    When m = n, Ly converges to Lx and P converges to identity mx.
    """
    args.Lx = torch.eye(args.m)*torch.abs(torch.randn((args.m, args.m)))*2  #utils.symmetrize(torch.randn((args.m, args.m)))
    args.m = 5
    args.n = 5
    args.Lx = torch.randn(args.m*(args.m-1)//2)  #torch.FloatTensor([[1, -1], [-1, 2]])
    args.Lx = graph.realize_upper(args.Lx, args.m)
    #args.Lx = torch.exp(torch.FloatTensor([[2, -2], [-2, 1]]))  #good initializations?! checks & stability
    args.n_epochs = 280
    graph.graph_dist(args)
    return

if __name__ == '__main__':
    """
    Driver class for various methods in this script. Can comment or uncomment depending on application.
    """
    args = utils.parse_args()
    #The following can be uncommented to run a particular method.
    #test_same_dim(args)
    #run_community_graph(args)
    #run_cyclic_graph(args)
    #run_FGW(args)
    #args.verbose = False
    scheme = 'lo_hi' #in format of query_data. 'hi_lo'
    if scheme == 'lo_hi':
        q_dim, data_dim = 20, 20 #100
        #q_dim, data_dim = 20, 30 #100
    else:
        q_dim, data_dim = 20, 10
    args.n_per_cls = 500 #200 #20 #15 #5 #15 # 20 #5
    #create_graphs(30, args, 'data/graphs{}.pkl'.format(30), n_graphs=args.n_per_cls) #do 30 to 10

    do_create_graph = False
    if do_create_graph:
        #runGraph.create_graphs(q_dim, args, 'data/queries{}rand.pkl'.format(q_dim), n_graphs=10, low=data_dim)
        runGraph.create_graphs(data_dim, args, 'data/train_graphs{}rand.pkl'.format(data_dim), n_graphs=args.n_per_cls, low=data_dim) #do 30 to 10
    do_generate_data = True
    if do_generate_data:
        dataset_name = args.dataset_name #e.g. 'IMDB-MULTI' 
        generate_real_data(dataset_name)

    lo_dim = 10 #15 #25 #15 #q_dim
    do_sketch_data = False
    if do_sketch_data: 
        #'''
        dataset, dataset_cls = utils.load_data('data/train_graphs{}rand.pkl'.format(data_dim))
        #dataset = dataset[10:-50]
        #dataset_cls = dataset_cls[10:-50]
        #dataset = dataset[:20]
        #dataset_cls = dataset_cls[:20]        
        lo_graphs, lo_cls = sketch_graph(dataset, dataset_cls, lo_dim, args)
        with open('data/train_graphs_sketch{}_{}rand.pkl'.format(data_dim,lo_dim), 'wb') as f:
            pickle.dump({'graphs':lo_graphs, 'labels':lo_cls}, f)
        '''
        #~~~#
        
        queries, target = utils.load_data('data/queries{}rand.pkl'.format(q_dim))
        lo_queries = sketch_graph(queries, lo_dim, args)
        with open('data/queries_sketch{}_{}rand.pkl'.format(q_dim, lo_dim), 'wb') as f:
            pickle.dump({'graphs':lo_queries, 'labels':target}, f)
        '''
        print('Done sketching graphs!')
        
    '''
    args.n_epochs = 150
    use_sketch = True #True #False #False #True #True
    dataset0, dataset_cls = utils.load_data('data/graphs{}rand.pkl'.format(data_dim))
    queries0, target = utils.load_data('data/queries{}rand.pkl'.format(q_dim))
    if use_sketch:
        #lo_dim = 10
        dataset, dataset_cls = utils.load_data('data/graphs_sketch{}_{}rand.pkl'.format(data_dim, lo_dim))        
        queries, target = utils.load_data('data/queries_sketch{}_{}rand.pkl'.format(q_dim, lo_dim))
    else:
        dataset, queries = dataset0, queries0
        #queries, target = utils.load_data('data/queries{}.pkl'.format(8))
    
    queries = queries[::3]
    target = target[::3]
    queries0 = queries0[::3]
    
    #classify_st(dataset, queries, dataset_cls, target, args, dataset0=dataset0, queries0=queries0)
    classify(dataset, queries, dataset_cls, target, args, dataset0=dataset0, queries0=queries0)
    '''
