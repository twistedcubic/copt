"""
Create and plot various COPT-based visualizations.
"""
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
import sys
import runGraph
import os
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import pdb

torch.set_default_tensor_type('torch.DoubleTensor')


def view2d(dataset_lo, dataset_cls_lo, args):
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = plt.gca()
    #ax.scatter(xs, ys, zs, c=c, marker=m)
    n_data = len(dataset_lo)
    Ly_mx = []
    ones = torch.ones(3, 3).triu(diagonal=1)
    for i, data in enumerate(dataset_lo):
        #n_nodes = len(data.nodes())
        L = utils.graph_to_lap(data)
        #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])        
        L = L[ones > 0][:2]       #.view(-1)
        L = torch.sort(L, -1)[0] #torch.topk(L,k=2,dim=-1)[0]
        Ly_mx.append(L)

    cls2c = {0:'r',1:'b',2:'g',3:'c',4:'m',5:'k',6:'y'}
    cls2label = {0:'block2',1:'rand_reg',2:'barabasi',3:'block3',4:'block4',5:'k', 6:'y'}
    c_l = []
    for i in range(n_data):
        c_l.append(cls2c[dataset_cls_lo[i]])
        
    ar = torch.stack(Ly_mx)
    #pdb.set_trace()
    #ar = (ar/torch.norm(ar, 2, dim=-1, keepdim=True) ).t().numpy() #.transpose() #.t().numpy()
    ar = (ar).t().numpy() #.transpose() #.t().numpy()
    ax.set_ylim(-7, ar[0].max())
    
    #
    #ax.set_zlim3d(-15, ar[1].max())
    #ax.set_xlim3d(-20, ar[2].max())
    
    ax.set_xlim(-25, ar[1].max())

    range_ar = np.array(list(range(n_data)))
    #ax.scatter(ar[1], ar[0], c=c_l)
    for i in range(5):
        #if i == 4:
        #    continue
        idx = range_ar[dataset_cls_lo==i]
        #pdb.set_trace()
        ax.scatter(ar[0][idx], ar[1][idx], c=cls2c[i], label=cls2label[i])#, c=c, marker=m)    

    #'''
    ax.legend()
    path = 'data/projection_2d{}.jpg'.format(args.data_dim)
    fig.savefig(path)
    print('plot saved to {}'.format(path))
    #'''
    
def view(dataset_lo, dataset_cls_lo, args):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(xs, ys, zs, c=c, marker=m)
    n_data = len(dataset_lo)
    Ly_mx = []
    ones = torch.ones(3, 3).triu(diagonal=1)
    for i, data in enumerate(dataset_lo):
        #n_nodes = len(data.nodes())
        L = utils.graph_to_lap(data)
        #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])        
        L = torch.sort(L[ones > 0].view(-1))[0]
        Ly_mx.append(L)

    #cls2label = {0:0, 1:1, 4:2, 5:3, 7:4, 8:5, 2:6, 3:7}
    #cls2label = {0:0, 1:1, 4:2, 5:3, 7:4, 8:5, 2:6, 3:7}
    #cls2c = {0:'r',1:'b',2:'g',3:'c',4:'m',5:'k',6:'y', 7:'.77'}
    cls2c = {0:'r',1:'b',2:'g',3:'c',4:'m',5:'k',6:'y', 7:'.77'}
    #cls2label = {0:'block2',1:'rand_reg',2:'barabasi',3:'block3',4:'block4',5:'k',6:'y',7:'4'}
    cls2label = {0:'block-2',1:'random regular',2:'barabasi',3:'block-3',4:'powerlaw tree',5:'caveman',6:'watts-strogatz',7:'binomial'}
    c_l = []
    labels = []
    for i in range(n_data):
        c_l.append(cls2c[dataset_cls_lo[i]])
        labels.append(cls2c[dataset_cls_lo[i]])
    ar = torch.stack(Ly_mx)
    #pdb.set_trace()
    #ar = (ar/torch.norm(ar, 2, dim=-1, keepdim=True)).t().numpy() #.transpose() #.t().numpy()
    ar = (ar).t().numpy() #.transpose() #.t().numpy()
    #ax.scatter(ar[0], ar[1], ar[2], c=c_l, label=labels)#, c=c, marker=m)
    ####
    #'''
    #zoom out
    ax.set_ylim3d(-17, ar[1].max())
    ax.set_zlim3d(-10, ar[2].max())
    ax.set_xlim3d(-20, ar[0].max())
    #'''
    #'''
    #zoom in
    ax.set_ylim3d(-2, ar[1].max())
    ax.set_zlim3d(-1, ar[2].max())
    ax.set_xlim3d(-5, ar[0].max())
    range_ar = np.array(list(range(n_data)))
    markers = ['^', 'o', 'x', '.', '1', '3', '+', '4', '5']
    marker_cnt = 0
    for i in range(8):
        if i == 6 or i == 7 or i == 2 or i == 3:
            continue
        idx = range_ar[dataset_cls_lo==i]
        #pdb.set_trace()
        #ax.scatter(ar[0][idx], ar[1][idx], ar[2][idx], c=cls2c[i], label=cls2label[i])#, c=c, marker=m)
        ax.scatter(ar[0][idx], ar[1][idx], ar[2][idx], c=cls2c[i], marker=markers[marker_cnt], label=cls2label[i])#, c=c, marker=m)
        marker_cnt += 1
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    #plt.title('3D sketches of 10-node graphs (zoomed)', fontsize=18)
    plt.title('3D sketches (zoomed)', fontsize=18)
    #plt.title('3D sketches of 10-node graphs', fontsize=18)
    #plt.title('Three dimensional COPT projections of 20-node graphs')
    #'''
    path = 'data/projection_{}.jpg'.format(args.data_dim)
    fig.savefig(path)
    print('plot saved to {}'.format(path))
    #'''


def create_and_sketch(data_dim, lo_dim, args, path, n_graphs): #do 30 to 10:

    if os.path.exists(path):
        with open(path, 'rb') as f:
            m = pickle.load(f)
            #m = {'graphs':graphs, 'sketch': lo_graphs, 'labels':labels}
        return m['graphs'], m['sketch'], m['labels'] 
    graphs, labels = runGraph.create_graphs(data_dim, args, path, n_graphs, low=data_dim, save=False)
    
    #dataset0, dataset_cls0, dataset_lo, dataset_cls_lo = create_and_sketch(data_dim, lo_dim, args, 'data/train_graphs{}dual{}_{}.pkl'.format(data_dim, lo_dim, args.n_per_cls), n_graphs=args.n_per_cls) #do 30 to 10
    
    lo_graphs = sketch_graph(graphs, lo_dim, args)
    with open(path, 'wb') as f:
        pickle.dump({'graphs':graphs, 'sketch': lo_graphs, 'labels':labels}, f)
    return graphs, lo_graphs, labels

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
    args = utils.parse_args()
    scheme = 'lo_hi' #in format of query_data. 'hi_lo'
    if scheme == 'lo_hi':
        q_dim, data_dim = 50, 10 #10 #50 #100
        #q_dim, data_dim = 20, 30 #100
    else:
        q_dim, data_dim = 20, 10
        
    args.q_dim, args.data_dim = q_dim, data_dim
    args.n_per_cls = 20 #30 #100 #20 #15 #5 #15 # 20 #5
    #create_graphs(30, args, 'data/graphs{}.pkl'.format(30), n_graphs=args.n_per_cls) #do 30 to 10    
    
    if False: #False: #True:#False: #True: #False: #True: #False: #False:#True:#True: #False: #False: #True: #False:   
        #create_graphs(q_dim, args, 'data/queries{}rand.pkl'.format(q_dim), n_graphs=10)
        create_graphs(data_dim, args, 'data/train_graphs{}rand.pkl'.format(data_dim), n_graphs=args.n_per_cls) #do 30 to 10
        
    args.lo_dim = 3 #15 #15 #q_dim
    lo_dim = args.lo_dim
    if False:#True: #True:  #False: #True: #False: #True: #True: #False: #False: #True:
        #'''
        dataset, dataset_cls = utils.load_data('data/train_graphs{}rand.pkl'.format(data_dim))
        lo_graphs = sketch_graph(dataset, lo_dim, args)
        with open('data/train_graphs_sketch{}_{}rand.pkl'.format(data_dim,lo_dim), 'wb') as f:
            pickle.dump({'graphs':lo_graphs, 'labels':dataset_cls}, f)
        '''
        #~~~#
        
        queries, target = utils.load_data('data/queries{}rand.pkl'.format(q_dim))
        lo_queries = sketch_graph(queries, lo_dim, args)
        with open('data/queries_sketch{}_{}rand.pkl'.format(q_dim, lo_dim), 'wb') as f:
            pickle.dump({'graphs':lo_queries, 'labels':target}, f)
        '''
        print('Done sketching graphs!')
        
    args.sketch = False #True #False #True
    if args.sketch:
        #dataset0, dataset_cls0 = utils.load_data('data/train_graphs_sketch{}_{}randVar.pkl'.format(data_dim, lo_dim))
        dataset0, dataset_cls0 = utils.load_data('data/train_graphs_sketch{}_{}rand.pkl'.format(data_dim, lo_dim))
        #dataset0.extend(dataset00)
        #dataset_cls0.extend(dataset_cls00)
        #idx = torch.random.randint() ###
        idx = np.random.permutation(len(dataset0))#.tolist()
        dataset1, dataset_cls1 = [dataset0[i] for i in idx], [dataset_cls0[i] for i in idx]
         
        #dataset0, dataset_cls0 = dataset0[idx], dataset_cls0[idx]
        dataset, dataset_cls = dataset1[50:], dataset_cls1[50:]
        val_set, val_cls = dataset1[:50], dataset_cls1[:50]
        
    else:
        #dataset_lo, dataset_cls_lo = utils.load_data('data/train_graphs_sketch{}_{}rand.pkl'.format(data_dim, lo_dim))
        #dataset0, dataset_cls0 = utils.load_data('data/train_graphs{}rand.pkl'.format(data_dim))
        
        dataset0, dataset_lo, dataset_cls0 = create_and_sketch(data_dim, lo_dim, args, 'data/train_graphs{}dual{}_{}.pkl'.format(data_dim, lo_dim, args.n_per_cls), n_graphs=args.n_per_cls) #do 30 to 10
        dataset_cls_lo = dataset_cls0
        
        #pdb.set_trace()
        #dataset_lo, dataset_cls_lo = s
        np.random.seed(0)
        idx = np.random.permutation(len(dataset0))#.tolist()
        dataset1, dataset_cls1 = [dataset0[i] for i in idx], [dataset_cls0[i] for i in idx]
        dataset_lo1, dataset_cls_lo1 = [dataset_lo[i] for i in idx], [dataset_cls_lo[i] for i in idx]
        n_val = 180 #20
        n_data = 600 #200 it's .4
        dataset, dataset_cls = dataset1[n_val:n_val+n_data], dataset_cls1[n_val:n_val+n_data]
        val_set, val_cls = dataset1[:n_val], dataset_cls1[:n_val]
        dataset_lo, dataset_cls_lo = dataset_lo1[n_val:n_val+n_data], dataset_cls_lo1[n_val:n_val+n_data]
        val_set_lo, val_cls_lo = dataset_lo1[:n_val], dataset_cls_lo1[:n_val]
        
        #val_set, val_cls = utils.load_data('data/queries{}rand.pkl'.format(data_dim))
    #'''
    #cls2label = {0:0, 1:1, 4:2, 5:3, 6:4, 9:5}
    
    cls2label = {0:0, 1:1, 4:2, 5:3, 7:4, 8:5, 2:6, 3:7}
    labels = []
    dataset_cls = np.array([cls2label[c] for c in dataset_cls])
    val_cls = np.array([cls2label[c] for c in val_cls])
    val_cls_lo = np.array([cls2label[c] for c in val_cls_lo])
    
    #'''
    

    args.k = 30 #40 #24 #100 #@50 #50 #100
    #view2d(val_set_lo, val_cls_lo, args)
    view(val_set_lo, val_cls_lo, args)
    
