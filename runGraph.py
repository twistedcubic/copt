
"""
Can be run e.g.:
python runGraph.py --hike --hike_interval 10 --dataset_type real --dataset_name PROTEINS
python runGraph.py --hike --hike_interval 10 --dataset_type synthetic
"""

import _init_paths
import utils
import graph

import networkx as nx
import torch

try:
    import lib.graph as gwGraph
    from lib.ot_distances import Fused_Gromov_Wasserstein_distance
except:
    print('NOTE: GW library not found. This is not required. Clone the GW repo as in README if running GW baseline')
    
import numpy as np
import pickle
from tqdm import tqdm
import got_stochastic as st
import sys
import time
import sklearn.model_selection
import netlsd

import collections

import pdb

torch.set_default_tensor_type('torch.DoubleTensor')

def classify_st(dataset, queries, dataset_cls, target, args, dataset0=None, queries0=None):
    """
    classify graphs. Can be used to compare COPT, GOT.
    Input: dataset, queries: could be sketched or non-sketched.
    dataset0, queries0: original, non-sketched graphs.
    """
    
    if dataset0 is None:
        dataset0 = dataset
        queries0 = queries    
    n_data = len(dataset)
    n_queries = len(queries)
    ot_cost = np.zeros((len(queries), len(dataset)))
    
    st_cost = np.zeros((len(queries), len(dataset)))
    
    Ly_mx = []
    Lx_mx = []
    data_graphs = []    
    
    for i, data in enumerate(dataset):
        n_nodes = len(data.nodes())
        L = utils.graph_to_lap(data)
        #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])
        Ly_mx.append(L)

    #pdb.set_trace()
    for i, q in enumerate(tqdm(queries, desc='queries')):
        Lx = utils.graph_to_lap(q)
        args.Lx = Lx
        
        args.m = len(q.nodes())
        Lx_mx.append(args.Lx)
        n_repeat = 1 #1 works fine
        for j, data in enumerate(dataset):
            
            Ly = Ly_mx[j].clone()
            args.n = len(Ly)
            min_loss = 10000
            
            for _ in range(n_repeat):
                loss, P, Ly_ = graph.graph_dist(args, plot=False, Ly=Ly, take_ly_exp=False)                
                if loss < min_loss:
                    min_loss = loss                
            
            ot_cost[i][j] = min_loss
            try:
                x_reg, y_reg, (P_st, loss_st) = st.find_permutation(Lx.cpu().numpy(), Ly.cpu().numpy(), args.st_it, args.st_tau, args.st_n_samples, args.st_epochs, args.st_lr, loss_type = 'w', alpha = 0, ones = True, graphs = True) #l2
            except Exception:
                print('Exception encountered during GOT')
                #pdb.set_trace()
                
            st_cost[i][j] = loss_st            
          
    ##can also try median, or dataset_cls[np.argsort(ot_cost[-8],-1)[:10]], or  dataset_cls[np.argpartition(ot_cost[6],10)[:10]]
    ot_cost_ = torch.from_numpy(ot_cost)
    #for combined, can add dist here
    ot_cost_ranks = torch.argsort(ot_cost_, -1)[:, :args.n_per_cls]
    ones = torch.ones(100)  #args.n_per_cls*2 (n_cls*2)
    ot_cls = np.ones(n_queries)
    
    dataset_cls_t = torch.from_numpy(dataset_cls)
    
    for i in range(n_queries): #for each cls 
        cur_ranks = dataset_cls_t[ot_cost_ranks[i]]
        ranked = torch.zeros(100) #n_cls*2
        ranked.scatter_add_(src=ones, index=cur_ranks, dim=-1)
        ot_cls[i] = torch.argmax(ranked).item()
                
    ot_cost_means = np.mean(ot_cost.reshape(n_queries, n_data//args.n_per_cls, args.n_per_cls), axis=-1)
    ot_idx = np.argmin(ot_cost_means, axis=-1) * args.n_per_cls
    
    st_cost_means = np.mean(st_cost.reshape(n_queries, n_data//args.n_per_cls, args.n_per_cls), axis=-1)
    st_idx = np.argmin(st_cost_means, axis=-1) * args.n_per_cls
    
    ot_cls1 = dataset_cls[ot_idx]
    
    st_cls = dataset_cls[st_idx]
    ot_acc, ot_acc1 = np.equal(ot_cls, target).sum() / len(target), np.equal(ot_cls1, target).sum() / len(target)
    st_acc = np.equal(st_cls, target).sum() / len(target)
    

    print('ot acc1 {} ot acc {} st acc {}'.format(ot_acc1, ot_acc, st_acc))

    return


def classify(dataset, queries, dataset_cls, target, args, dataset0=None, queries0=None):
    """
    classification tasks using various methods.
    dataset0, queries0 are original, non-sketched graphs. dataset, queries contain sketched graphs.
    """
    if dataset0 is None:
        dataset0 = dataset
        queries0 = queries    
    #with open(args.graph_fname, 'rb') as f:
    #    graphs = pickle.read(f)
    n_data = len(dataset)
    n_queries = len(queries)
    ot_cost = np.zeros((len(queries), len(dataset)))
    
    netlsd_cost = np.zeros((len(queries), len(dataset)))
    
    Ly_mx = []
    Lx_mx = []
    data_graphs = []
    heat_l = []
    #avg_deg = 0
    for i, data in enumerate(dataset):
        #pdb.set_trace()
        if isinstance(data, torch.Tensor):
            L = data
        else:
            n_nodes = len(data.nodes())
            L = utils.graph_to_lap(data)
            
        avg_deg = (L.diag().mean())
        L /= avg_deg
        #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])
        Ly_mx.append(L)
        #pdb.set_trace()
        
        heat_l.append(netlsd.heat(L.numpy())) 
    #avg_deg /= len(dataset)
    
    for i, q in enumerate(tqdm(queries, desc='queries')):
        '''###
        if isinstance(data, torch.Tensor):
            L = data
        else:
            n_nodes = len(data.nodes())
            L = utils.graph_to_lap(data)

        '''
        Lx = utils.graph_to_lap(q) 
        avg_deg = (Lx.diag().mean())
        Lx /= avg_deg
      
        args.Lx = Lx
        
        
        args.m = len(q.nodes())
        q_heat = netlsd.heat(Lx.numpy()) 
        Lx_mx.append(args.Lx)
        
        for j, data in enumerate(dataset):
            
            Ly = Ly_mx[j].clone()            
            args.n = len(Ly)
            min_loss = 10000
            
            for _ in range(1):
                loss, P, Ly_ = graph.graph_dist(args, plot=False, Ly=Ly, take_ly_exp=False)
                #pdb.set_trace()
                if loss < min_loss:
                    min_loss = loss            
            
            ot_cost[i][j] = min_loss
            netlsd_cost[i][j] = netlsd.compare(q_heat, heat_l[j])
                    
    if args.dataset_type == 'real':
        ot_cost1 = (ot_cost-ot_cost.mean())/np.std(ot_cost)
        ot_pred = ot_cost.argmin(1)
        ot_acc00 = np.equal(dataset_cls[ot_pred],target).sum()/len(target)
        
        print('OT ACC |{} '.format(ot_acc00))
                
        ot_sorted = np.argsort(ot_cost, axis=-1)
        
        #pdb.set_trace()
        ot_cls = dataset_cls[ot_sorted[:, :3]].tolist()
        
        combine_pred = np.zeros(len(target))
        for i, ot_c in enumerate(ot_cls):
                        
            counter = collections.Counter()
            counter.update(ot_c)
            
            #pdb.set_trace()
            common = counter.most_common(1)[0][0]
            combine_pred[i] = common
            
        combine_acc = np.equal(combine_pred,target).sum()/len(target)
        #pdb.set_trace()
        ###
        ot_pred = ot_cost.argmin(1)
        ot_acc = np.equal(dataset_cls[ot_pred],target).sum()/len(target)
        
        netlsd_pred = netlsd_cost.argmin(1)
        netlsd_acc = np.equal(dataset_cls[netlsd_pred],target).sum()/len(target)
        print('OT ACC |{} '.format(ot_acc))
        return ot_acc00, netlsd_acc
    
    ot_cost_ = torch.from_numpy(ot_cost)
    #for combined, can add dist here
    ot_cost_ranks = torch.argsort(ot_cost_, -1)[:, :args.n_per_cls]
    ones = torch.ones(args.n_per_cls*3)  #args.n_per_cls*2 (n_cls*2) 100
    ot_cls = np.ones(n_queries)
    
    combine_cls = np.ones(n_queries)
    dataset_cls_t = torch.from_numpy(dataset_cls)
    #pdb.set_trace()
    for i in range(n_queries): #for each cls 
        cur_ranks_ot = dataset_cls_t[ot_cost_ranks[i]]
        ranked = torch.zeros(100) #n_cls*2
        ranked.scatter_add_(src=ones, index=cur_ranks_ot, dim=-1)
        ot_cls[i] = torch.argmax(ranked).item()
        
    ot_cost_means = np.mean(ot_cost.reshape(n_queries, n_data//args.n_per_cls, args.n_per_cls), axis=-1)
    ot_idx = np.argmin(ot_cost_means, axis=-1) * args.n_per_cls
    
    print('ot_cost mx ', ot_cost)
    ot_cls1 = dataset_cls[ot_idx]     
    ot_acc, ot_acc1 = np.equal(ot_cls, target).sum() / len(target), np.equal(ot_cls1, target).sum() / len(target)
    print('ot acc1 {} ot acc {} '.format(ot_acc1, ot_acc))


def run_perm_mi(args):
    n_repeat = 20  
    for n in range(4,5): #range(5,6): #range(7, 9):
        for _ in range(n_repeat):
            args.n_remove = 30*n
            perm_mi(args)
    
def perm_mi(args):
    '''
    Remove edges, permute, align, then measure MI.
    '''
    args.n_epochs = 1000
    params = {'n_blocks':4}
    use_given_graph = False
    if use_given_graph: #True:#False: #True:
        g = torch.load('mi_g_.pt')
    else:
        seed = 0 if args.fix_seed else None
        g = utils.create_graph(40, gtype='block', params=params, seed=seed)
        #torch.save(g, 'mi_g.pt')
    orig_cls = []
    for i in range(4):
        orig_cls.extend([i for _ in range(10)])
    orig_cls = np.array(orig_cls)    
    Lg = utils.graph_to_lap(g)
    args.Lx = Lg.clone()
    args.m = len(Lg)
    
    #remove edges and permute
    n_remove = args.n_remove #150
    rand_seed = 0 if args.fix_seed else None
    Lg_removed = utils.remove_edges(Lg, n_remove=n_remove, seed=rand_seed)    
    Lg_perm, perm = utils.permute_nodes(Lg_removed.numpy(), seed=rand_seed)
    
    inv_perm = np.empty(args.m, perm.dtype)
    inv_perm[perm] = np.arange(args.m)
    
    ##Ly = torch.from_numpy(Lg_perm)
    Ly = torch.from_numpy(Lg_perm) #Lg_removed.clone() #args.Lx.clone()
    args.n = len(Ly)
    #8 st_n_samples worked best, 5 sinkhorn iter, 1 as tau
    #align
    time0 = time.time()
    loss, P, Ly_ = graph.graph_dist(args, plot=False, Ly=Ly, take_ly_exp=False)
    dur_ot = time.time() - time0
    
    orig_idx = P.argmax(-1).cpu().numpy()
    perm_mx = False
    if perm_mx:
        P_max = P.max(-1, keepdim=True)[0]        
        P[P<P_max-.1] = 0
        P[P > 0] = 1
    
    new_cls = orig_cls[perm][orig_idx].reshape(-1)         
    mi = utils.normalizedMI(orig_cls, new_cls)
    #return mi
    Lx = args.Lx
    time0 = time.time()
    x_reg, y_reg, (P_st, loss_st) = st.find_permutation(Ly.cpu().numpy(), Lx.cpu().numpy(), args.st_it, args.st_tau, args.st_n_samples, args.st_epochs, args.st_lr, loss_type = 'w', alpha = 0, ones = True, graphs = True) 
    dur_st = time.time() - time0    
    orig_idx = P_st.argmax(-1)

    new_cls_st = orig_cls[perm][orig_idx].reshape(-1)    
    mi_st = utils.normalizedMI(orig_cls, new_cls_st)
    #print('{} COPT {} GOT {} dur ot {} dur st {}'.format(n_remove, mi, mi_st, dur_ot, dur_st))
    print('{} {} {} {} {}'.format(n_remove, mi, mi_st, dur_ot, dur_st))    
    return mi
    
def create_graphs(n_, args, fname, n_graphs, low=None, save=True):
    """
    Create graphs
    """
    labels = []
    graphs = []
    #create ran
    low = max(int(n_//1.4), 25) if low is None else low
    n_l = np.random.randint(low, high=n_+1, size=n_graphs*15) ##
    labels.extend([0]*n_graphs)
    for i in range(n_graphs):
        params = {'n_blocks':2}
        n = n_l[0*n_graphs + i]
        graphs.append(utils.create_graph(n, 'block', params=params))
    labels.extend([1]*n_graphs)
    for i in range(n_graphs):
        #sketching dist ~5e2
        n = n_l[1*n_graphs + i]
        graphs.append(utils.create_graph(n, 'random_regular'))

    labels.extend([4]*n_graphs)
    for i in range(n_graphs):
        n = n_l[4*n_graphs + i]
        graphs.append(utils.create_graph(n, 'barabasi'))
    labels.extend([5]*n_graphs)
    for i in range(n_graphs):
        params = {'n_blocks':3}
        n = n_ #n_l[5*n_graphs + i]
        graphs.append(utils.create_graph(n, 'block', params=params))
    labels.extend([6]*n_graphs)
    for i in range(n_graphs):
        params = {'n_blocks':4}
        n = n_ #n_l[6*n_graphs + i]
        #2 and 6 confused
        graphs.append(utils.create_graph(n, 'block', params=params))
    labels.extend([9]*n_graphs)
    for i in range(n_graphs):
        params = {'radius': .2} #, 'clique_sz':7}
        n = n_l[9*n_graphs + i]
        graphs.append(utils.create_graph(n, 'random_geometric', params=params))

    if save:
        with open(fname, 'wb') as f:
            pickle.dump({'graphs':graphs, 'labels':np.array(labels)}, f)
    #save graphs
    return graphs, np.array(labels)

def create_graphs_view(n_, args, fname, n_graphs, low=None, save=True):
    """
    classify graphs
    """
    labels = []
    graphs = []
    #create ran
    low = max(int(n_//1.4), 25) if low is None else low
    n_l = np.random.randint(low, high=n_+1, size=n_graphs*15) ##
    labels.extend([0]*n_graphs)
    for i in range(n_graphs):
        params = {'n_blocks':2}
        n = n_l[0*n_graphs + i]
        graphs.append(utils.create_graph(n, 'block', params=params))
    labels.extend([1]*n_graphs)
    for i in range(n_graphs):
        #sketching dist ~5e2
        n = n_l[1*n_graphs + i]
        graphs.append(utils.create_graph(n, 'random_regular'))
    #'''
    labels.extend([2]*n_graphs)
    #confused with 1?
    for i in range(n_graphs):
        n = n_l[2*n_graphs + i]
        graphs.append(utils.create_graph(n, 'strogatz'))
    labels.extend([3]*n_graphs)    
    for i in range(n_graphs):
        params = {'prob':.2}
        graphs.append(utils.create_graph(n, 'binomial', params=params))
    
    labels.extend([4]*n_graphs)
    for i in range(n_graphs):
        n = n_l[4*n_graphs + i]
        graphs.append(utils.create_graph(n, 'barabasi'))
    labels.extend([7]*n_graphs)
    for i in range(n_graphs):      
        graphs.append(utils.create_graph(n, 'powerlaw_tree'))
    labels.extend([8]*n_graphs)    
    for i in range(n_graphs):
    #all close to
        #ideal lr .05, can learn but large loss
        params = {'n_cliques':max(1,n//7), 'clique_sz':7}
        graphs.append(utils.create_graph(n//7*7, 'caveman', params=params))
    if save:
        with open(fname, 'wb') as f:
            pickle.dump({'graphs':graphs, 'labels':np.array(labels)}, f)
    #save graphs
    return graphs, np.array(labels)

def sketch_graph(graphs, lo_dim, args):
    '''
    Run graph sketching.
    Input: graphs: graphs to be dimension-reduced for..
    '''
    args.n = lo_dim
    lo_graphs = []
    args.n_epochs = 230
    for g in tqdm(graphs, desc='sketching'):
        args.Lx = utils.graph_to_lap(g) #graph.graph_dist(args, plot=False)        
        args.m = len(args.Lx)
        #sys.stdout.write(' ' +str(len(g.nodes())))
        #sys.stdout.write(str(args.m) +' ')        
        loss, P, Ly = graph.graph_dist(args, plot=False)
        
        lo_graphs.append(utils.lap_to_graph(Ly))
        
    return lo_graphs

def test_FGW(args):
    """
    Sample/test run Fused Gromov-Wasserstein distance.
    """
    args.m = 8
    args.n = 4
    if args.fix_seed:
        torch.manual_seed(0)
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
    
def run_cyclic_graph(args):
    """
    Test for sketching Cyclic graph, and some other sanity checks. 
    """
    #e.g. test if Lx and Ly are the same, then dist is very small
    args.m = 8
    args.n = 4
    '''
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
    '''
    ones = torch.ones((args.m, args.m), dtype=torch.uint8)
    Lx = torch.zeros((args.m, args.m))
    Lx[torch.triu(ones, diagonal=1) & torch.tril(ones, diagonal=1)] = -1
    Lx[0, -1] = -1
    Lx += Lx.t()    
    Lx[torch.eye(args.m) > 0] = -Lx.sum(0)
    args.Lx = Lx
    args.n_epochs = 370 #370
    graph.graph_dist(args)
        

def run_community_graph(args):
    #test if Lx and Ly are the same, then dist should be small!
    #laplacian make integral at end.   Inverse is often quite small for images!  ?? leading to tiny evals. even neg
    #args.Lx = torch.eye(args.m)*torch.abs(torch.randn((args.m, args.m)))*2  #utils.symmetrize(torch.randn((args.m, args.m)))
    args.m = 12
    args.n = 4
    if args.fix_seed:
        torch.manual_seed(0)
    g = nx.stochastic_block_model([6,6],[[0.9,0.1],[0.1,0.9]], seed = 8576)
    #components = nx.connected_components(g)
    g.remove_nodes_from(list(nx.isolates(g)))
    args.m = len(g)    
    Lx = nx.laplacian_matrix(g, range(args.m)).todense()
    args.Lx = torch.from_numpy(Lx).to(dtype=torch.float32) #+ torch.ones(args.m, args.m)/args.m
    
    args.n_epochs = 370 #100
    graph.graph_dist(args)

def test_FGW(args):
    """
    Fused Gromov-Wasserstein distance
    """
    args.m = 8
    args.n = 4
    if args.fix_seed:
        torch.manual_seed(0)
    g = nx.stochastic_block_model([4,4],[[0.9,0.1],[0.1,0.9]], seed = 8576)
    #components = nx.connected_components(g)
    g.remove_nodes_from(list(nx.isolates(g)))
    args.m = len(g)
    g2 = nx.stochastic_block_model([4,4],[[0.9,0.1],[0.1,0.9]])
    #components = nx.connected_components(g)
    g2.remove_nodes_from(list(nx.isolates(g2)))
    args.n = len(g2)    
    gwdist = Fused_Gromov_Wasserstein_distance(alpha=0.8,features_metric='sqeuclidean')
    graph.graph_dist(args) 
    '''
    g = gwGraph.Graph(g)
    g2 = gwGraph.Graph(g2)
    dist = gwdist.graph_d(g,g2)    
    print('GW dist ', dist)
    '''
    pdb.set_trace()

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

def main_real_data(args):
    dataset_name = args.dataset_name #e.g. 'PROTEINS' 
    print('dataste name {}'.format(dataset_name))
    try:
        data, node_labels, cls = utils.fetch_data_graphs(dataset_name)
    except Exception:
        raise Exception('Dataset {} graph data not created yet. More data can be created using the generateData.py script as in README.'.format(dataset_name))
    #data, node_labels, cls = utils.fetch_data(dataset_name)
    #Can update the number of queries and data to classify against
    n_q = 10 #30 #30 #30    
    n_data = 30 #100 #100 #100
    print('number of query graphs {} number of data graphs {}'.format(n_q, n_data))
    n_runs = 1 # 10
    ot_acc_l = np.zeros(n_runs)
    gw_acc_l = np.zeros(n_runs)
    netlsd_acc_l = np.zeros(n_runs)
    combine_acc_l = np.zeros(n_runs)
    for i in range(n_runs):        
        #dataset0, queries0, dataset_cls, target = sklearn.model_selection.train_test_split(data, cls, test_size=.2, random_state=42*i)            
        dataset0, queries0, dataset_cls, target = sklearn.model_selection.train_test_split(data, cls, test_size=.2)

        dataset0, queries0, dataset_cls, target = dataset0[:n_data], queries0[:n_q], dataset_cls[:n_data], target[:n_q]            
        dataset, queries = dataset0, queries0

        ot_acc, netlsd_acc = classify(dataset, queries, dataset_cls, target, args, dataset0=dataset0, queries0=queries0)
        ot_acc_l[i] = ot_acc        
        netlsd_acc_l[i] = netlsd_acc
        
        print('SO FAR mean ot acc ({}) ot std {} netlsd acc {} netlsd std {} '.format(np.mean(ot_acc_l[:i+1]), np.std(ot_acc_l[:i+1]), np.mean(netlsd_acc_l[:i+1]), np.std(netlsd_acc_l[:i+1]), ))

    #print('mean ot acc {} ot std {} gw acc {} gw std {} netlsd acc {} netlsd std {}'.format(np.mean(ot_acc_l), np.std(ot_acc_l), np.mean(gw_acc_l), np.std(gw_acc_l), np.mean(netlsd_acc_l), np.std(netlsd_acc_l) ))
    return dataset0, queries0, dataset_cls, target
    
if __name__ == '__main__':
    args = utils.parse_args()
    if args.fix_seed:
        np.random.seed(42)
        torch.manual_seed(42)
        
    if not args.lr_hike:
        print('NOTE: for best (and faster) results, consider settings args.hike to True. Done by running with "--hike" argument')
        
    args.n_epochs = 340 
    #whether to compress graph
    #use_sketch = False #True #True #False #False #True #True
    use_sketch = True    
        
    if args.dataset_type == 'real':
        dataset0, queries0, dataset_cls, target = main_real_data(args)
    else:
        #synthetic data
        scheme = 'lo_hi' #in format of query_data. 'hi_lo'
        if scheme == 'lo_hi':
            #q stands for query
            q_dim, data_dim = 20, 20 #50 #100
            #q_dim, data_dim = 20, 30 #100
        else:
            q_dim, data_dim = 20, 10

        args.n_per_cls = 1 #15 
        args.query_n_per_cls = 1 #15 # set to 1 for testing runtime
        #use command line to set GOT and GW parameters, sample settings: args.st_it, args.st_tau, args.st_n_samples, args.st_epochs, args.st_lr = 5, 1, 10, 1000, .5    
        #generate_graph = False
        generate_graph = True
        if generate_graph: 
            create_graphs(q_dim, args, 'data/queries{}rand.pkl'.format(q_dim), n_graphs=args.query_n_per_cls, low=q_dim) 
            create_graphs(data_dim, args, 'data/graphs{}rand.pkl'.format(data_dim), n_graphs=args.n_per_cls, low=data_dim) 
        
        #lo_dim = 15 #q_dim
        args.lo_dim = 15 #50 #15 #q_dim
        lo_dim = args.lo_dim
        #sketch_data = False 
        sketch_data = True
        if sketch_data: 
            #'''
            dataset, dataset_cls = utils.load_data('data/graphs{}rand.pkl'.format(data_dim))
            #dataset = dataset[10:-50]
            #dataset_cls = dataset_cls[10:-50]
            #dataset = dataset[:20]
            #dataset_cls = dataset_cls[:20]        
            lo_graphs = sketch_graph(dataset, lo_dim, args)
            with open('data/graphs_sketch{}_{}rand.pkl'.format(data_dim,lo_dim), 'wb') as f:
                pickle.dump({'graphs':lo_graphs, 'labels':dataset_cls}, f)
            #'''
            #~~~#
            queries, target = utils.load_data('data/queries{}rand.pkl'.format(q_dim))
            lo_queries = sketch_graph(queries, lo_dim, args)
            with open('data/queries_sketch{}_{}rand.pkl'.format(q_dim, lo_dim), 'wb') as f:
                pickle.dump({'graphs':lo_queries, 'labels':target}, f)
            print('Done sketching graphs!')

        dataset0, dataset_cls = utils.load_data('data/graphs{}rand.pkl'.format(data_dim))
        queries0, target = utils.load_data('data/queries{}rand.pkl'.format(q_dim))
        
    if args.dataset_type == 'synthetic':
        #lo_dim = 10
        dataset, dataset_cls = utils.load_data('data/graphs_sketch{}_{}rand.pkl'.format(data_dim, lo_dim))        
        queries, target = utils.load_data('data/queries_sketch{}_{}rand.pkl'.format(q_dim, lo_dim))
    else:
        dataset, queries = dataset0, queries0
        #queries, target = utils.load_data('data/queries{}.pkl'.format(8))
    '''
    n_skip = 6
    queries = queries[::n_skip]
    target = target[::n_skip]
    queries0 = queries0[::n_skip]
    '''
    print('queries n {} data n {}'.format(len(queries), len(dataset)))
    #Toggle to compute mutual information for community detection
    test_mi = True
    test_mi = False
    if test_mi: 
        #classify_st(queries, queries, target, target, args, dataset0=dataset0, queries0=queries0)
        run_perm_mi(args)
    else:
        print('Not testing mi')
    #Toggle this switch to compare with GOT
    #do_compare_got = True
    do_compare_got = False
    print('NOTE: can toggle between whether to compare with GOT. Currently this is set to {}'.format(do_compare_got))
    if do_compare_got:
        print('Compare with GOT')
        classify_st(dataset, queries, dataset_cls, target, args, dataset0=dataset0, queries0=queries0)
    else:
        classify(dataset, queries, dataset_cls, target, args, dataset0=dataset0, queries0=queries0)
    
