"""
Filter out candidates using vectors created from COPT and spectral methods. Then classify with GW or other methods on the reduced number of candidates.
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
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import GridSearchCV
import stochastic as st
import torch.nn as nn
import sys
import runGraph
import os
import math
import netlsd
from sklearn.pipeline import Pipeline
import grakel
from sklearn.svm import SVC
import pickle
import time

import pdb

#optional
torch.set_default_tensor_type('torch.DoubleTensor')


def classify_svm_var_len(dataset, data_idx, query_idx, args, node_labels, q_labels, dataset_cls=None, tgt_cls=None, data_sketch=None, C_opt=None):
    #  pdb.set_trace()
    assert data_sketch is not None    
    all_graphs = []
    #pdb.set_trace()
    
    min_label = min([l.min().item() for l in node_labels])
    max_label = max([l.max().item() for l in node_labels])
    #pdb.set_trace()
    for i, sketch in enumerate(data_sketch):
        adj = -reconstruct_mx(sketch)
                
        n_nodes = len(adj)
        eye = torch.eye(n_nodes)
        adj /= adj.diag().sum().abs()        
        adj[eye>0] = 0
        adj = adj.numpy()        
        ###
        #adj = np.eye(n_nodes)
        ###
        
        #adj_l.append(adj)
        #node2attr = {node:lap[node] for node in range(len(adj))}
        #node2attr = [lap[node] for node in range(len(adj))]
        
        cur_node_labels = node_labels[i].numpy()
        node2attr = {node:cur_node_labels[node] for node in range(n_nodes)}
        all_graphs.append([adj, node2attr])

    train_graphs = [all_graphs[i] for i in data_idx] #[torch.from_numpy(data_idx)]
    test_graphs = [all_graphs[i] for i in query_idx] #[torch.from_numpy(query_idx)]
    train_cls = np.array([dataset_cls[i] for i in data_idx])
    L = 2
    if L != 3:
        sys.stdout.write('L! {}'.format( L))
    #pdb.set_trace()
    do_grid_search = args.grid_search #False #True #False
    
    if C_opt is None and do_grid_search:
        print('Doing grid search!')
        time0 = time.time()
        gk = grakel.GraphKernel(kernel=[{"name": "multiscale_laplacian", 'which':'fast', 'L':L, 'n_samples':100}], normalize=True)
        C_opt = grid_search(gk, train_graphs, train_cls)
        grid_dur = np.round((time.time()-time0)/60, 2)
        sys.stdout.write('C_opt found {} dur {} '.format(C_opt, grid_dur))
        #for train_idx, train_size in enumerate([0.7]):
        #average_acc, std_acc, average_time = ComputeAccuracy.ComputeAccuracy(all_graphs, dataset_cls, test_size=round(1-train_size,1), n_runs=2, cv_folds = 5) #n_runs=5
        gk = grakel.GraphKernel(kernel=[{"name": "multiscale_laplacian", 'L':L}], normalize=True) ##normalize!?
    else:
        C_opt = 1 if C_opt is None else C_opt
        use_lap = True #False
        if use_lap:
            if args.fast:
                #try to find best C using subset of data
                #gk = grakel.GraphKernel(kernel=[{"name": "multiscale_laplacian_fast", 'L':L, 'n_samples':150}], normalize=True)
                gk = grakel.GraphKernel(kernel=[{"name": "multiscale_laplacian", 'which':'fast', 'L':L, 'n_samples':100}], normalize=True)
                #using MultiscaleLaplacianFast is actually very slow
                #gk = grakel.MultiscaleLaplacianFast(L=L, normalize=True, n_samples=150)
            else:
                gk = grakel.GraphKernel(kernel=[{"name": "multiscale_laplacian", 'L':L}], normalize=True) ##normalize!?
            #
            #gk = grakel.MultiscaleLaplacian(L=3, normalize=True)
        else:
            gk = grakel.GraphKernel(kernel=[{"name": "pyramid_match"}], normalize=True) ##normalize!?
            G2 = []
            for g in all_graphs:
                node2idx = {}
                for k, v in g[1].items():
                    #pdb.set_trace()
                    node2idx[k] = np.argmax(v)
                g[1] = node2idx
                G2.append(g)
                all_graphs = G2
                train_graphs = [all_graphs[i] for i in data_idx] #[torch.from_numpy(data_idx)]
                test_graphs = [all_graphs[i] for i in query_idx] #[torch.from_numpy(query_idx)]
    time0 = time.time()
    k_train = gk.fit_transform(train_graphs)
    k_test = gk.transform(test_graphs)
    #clf = SVC(kernel='precomputed') #, C=10)
    clf = SVC(kernel='precomputed', C=C_opt)
    
    clf.fit(k_train, train_cls)
    fit_dur = np.round((time.time()-time0)/60, 2)
    sys.stdout.write(' time for actually fitting {}'.format(fit_dur))
    pred = clf.predict(k_test)
    acc = np.equal(pred.reshape(-1), tgt_cls).sum() / len(k_test)

    accs = acc
    print('cls acc {}'.format(accs))
    return accs, data_sketch, C_opt

def grid_search(gk, train_graphs, y_train):
    
    #k_test = gk.transform(test_graphs)
    n_total = min(150, len(train_graphs))
    n_train = int(n_total*.33)
    train_g = train_graphs[:n_train]
    train_cls = y_train[:n_train]
    test_g = train_graphs[n_train:n_total]
    test_cls = y_train[n_train:n_total]
    train_g = gk.fit_transform(train_g)
    test_g = gk.transform(test_g)
    C_l = [1, 10]
    accs = np.zeros(len(C_l))
    for i, cur_C in enumerate(C_l): 
        clf = SVC(kernel='precomputed', C=cur_C)        
        clf.fit(train_g, train_cls)
        pred = clf.predict(test_g)
        acc = np.equal(pred.reshape(-1), test_cls).sum() / len(test_cls)
        accs[i] = acc
    sys.stdout.write('acc for various C {}'.format(accs))
    C_opt = C_l[accs.argmax()]
    return C_opt

def grid_search_slow(gk, G_train, y_train):

    svc = SVC(kernel='precomputed')
    param_grid = {'svc__C': [1.5, 10]}
    estimator = Pipeline([('kernel', gk), ('svc',svc)])
    clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=2)
    #print(clf.get_params().keys()) 
    clf.fit(G_train, y_train)
    C_best = clf.best_estimator_.named_steps['svc'].C
    return C_best
    
def cross_validate(all_graphs, tgt):
    kernels = []
    for i in range(2, 7):
        gk = grakel.GraphKernel(kernel=[{"name": "multiscale_laplacian", 'n_iter':i}],  normalize=True) ##normalize!?
        kernel = gk.fit_transform()
        kernels.append(kernel)
    k_fold = 3
    accs = grakel.utils.cross_validate_Kfold_SVM([kernels], tgt, n_iter=10, n_splits=k_fold)
    print('{} fold cross validation accuracies {}'.format(k_fold, accs))

def reconstruct_mx(upper):
    #reconstruct full mx from upper triangular part
    n_nodes = int(-.5 + np.sqrt(.25 + 2*len(upper)))
    #
    mx = torch.zeros((n_nodes, n_nodes))
    ones = torch.ones((n_nodes, n_nodes))
    try:
        mx[ones.triu()>0] = upper
    except Exception:
        print('exception!')
        pdb.set_trace()    
    mx += mx.triu(diagonal=1).t()
    return mx

def classify_graphs(dataset, queries, args, node_labels, q_labels, dataset_cls=None, tgt_cls=None):#dataset_cls, target, args):
    """
    Initial dataset filtering to filter out candidates for GW distance. 
    dataset: dataset of laplacians, not sketched yet
    """
    print('using netlsd')
    #with open(args.graph_fname, 'rb') as f:
    #    graphs = pickle.read(f)
    n_data = len(dataset)
    n_queries = len(queries)
    #ot_cost = np.zeros((len(queries), len(dataset)))
    gw_cost = np.zeros((len(queries), len(dataset)))
    #gwdist = Fused_Gromov_Wasserstein_distance(alpha=args.alpha,features_metric='sqeuclidean')
    Ly_mx = []
    Lx_mx = []
    data_graphs = []
    Ly_n = []
    dataset_cls1 = []
    labels = []
    heat_l = []
    for i, data in enumerate(tqdm(dataset)):
        n_nodes = len(data)
        #assert n_nodes >= 10        
        #L = utils.graph_to_lap(data)
        #sketch
        #args.Lx = utils.graph_to_lap(g) #graph.graph_dist(args, plot=False)
        args.Lx = data
        args.m = len(args.Lx)
        args.n = math.ceil(args.m/args.compress_fac)
        loss, P, Ly = graph.graph_dist(args, plot=False)
        #pdb.set_trace()
        #pdb.set_trace()
        #lo_graphs.append(Ly)
        
        if False: #True: #False:
            cur_labels = node_labels[i][P.argmax(0)]
            #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])
            #order node labels
            
            L_ordering = Ly.diag().argsort(descending=True)
            cur_labels = cur_labels[L_ordering] ####
            labels.append(cur_labels)
        L = utils.canonicalize_mx(Ly)        
        heat_l.append(netlsd.heat(L.numpy()))
        '''
        L = L[ones > 0]        
        Ly_n.append( (L**2).sum())#.sqrt())
        Ly_mx.append(L)
        '''
    cur_ranks_ = torch.zeros(n_queries, args.k, dtype=torch.int64)
    #pdb.set_trace()
    
    for i, q in enumerate(tqdm(queries, desc='queries')):
        #Lx = utils.graph_to_lap(q)
        args.Lx = q
        args.m = len(q)
        args.n = math.ceil(args.m/args.compress_fac) #args.lo_dim #
        loss, P, Lx = graph.graph_dist(args, plot=False)        
        
        #q_graph = gwGraph.Graph(queries0[i])        
        #Lx_mx.append(args.Lx)
        Lx_n = (args.Lx**2).sum() #.sqrt()
        if False: #True: #False:
            #pdb.set_trace()
            cur_labels = q_labels[i][P.argmax(0)]
            #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])
            #order node labels
            #pdb.set_trace()
            L_ordering = Lx.diag().argsort(dim=0)
            q_label = cur_labels[L_ordering] ###
            
        Lx = utils.canonicalize_mx(Lx) #[ones > 0]
        ##Lx = torch.cat((Lx, q_label*.03), -1)
        #pdb.set_trace()
        #dist = Lx_n + Ly_n - 2*torch.mm(Lx.view(1, -1), data_t.t()).view(-1)
        #pdb.set_trace()
        #dist = torch.abs(Lx.view(1,-1) - data_t).sum(-1)
        
        #'''
        dist = []
        heat_q = netlsd.heat(Lx.numpy())
        for j,d in enumerate(heat_l):
            #dist.append(torch.norm(Lx-d, 1) )
            dist.append(netlsd.compare(heat_q, heat_l[j]))
        dist = torch.Tensor(dist)
        #'''
        
        cur_ranks_[i] = torch.topk(dist, args.k, largest=False)[1] #np.argpartition(dist, args.k)[:args.k]
        #cur_cls = dataset_cls[cur_ranks_[i]]
        #pdb.set_trace()
        
    if dataset_cls is not None:
        pdb.set_trace()
        pred = dataset_cls[cur_ranks_[:, 0].numpy()]        
        acc1 = np.equal(pred, tgt_cls).sum()
        print('top 1 acc using just mx dist ! {}'.format(acc1/n_queries))
        # top-ten voting #
        dataset_cls = torch.from_numpy(dataset_cls)        
        ones = torch.ones(50)
        pred10 = np.zeros(n_queries)
        for i, q in enumerate(queries):
            #cur_ranks = dataset_cls_t[ot_cost_ranks[i]]
            ranked = torch.zeros(100) #n_cls*2
            cur_ranks = dataset_cls[cur_ranks_[i, :30]]
            ranked.scatter_add_(src=ones, index=cur_ranks, dim=-1)
            #pdb.set_trace()
            pred10[i] = torch.argmax(ranked).item()            
        acc10 = np.equal(pred, tgt_cls).sum()
        print('top 30 voting acc using just mx dist ! {}'.format(acc10/n_queries))        
        
    return cur_ranks_

def classify_l1_var_len(dataset, data_idx, query_idx, args, node_labels, q_labels, dataset_cls=None, tgt_cls=None, data_sketch=None):
    """
    Initial dataset filtering to filter out candidates for GW distance. 
    dataset: dataset of laplacians, not sketched yet
    """
    print('using sketching l1 var len')
    n_data = len(data_idx)
    n_queries = len(query_idx)
    max_len = max([len(l) for l in dataset])
    #max_len = (max_len//5+1)*max_len//5 //2
    max_len = (math.ceil(max_len/args.compress_fac)+1)*math.ceil(max_len/args.compress_fac) //2
    Lx_mx = []
    data_graphs = []
    Ly_n = []
    dataset_cls1 = []
    labels = []
    #heat_l = []
    if data_sketch is None:
        data_sketch = []    
        for i, data in enumerate(tqdm(dataset)):
            n_nodes = len(data)        
            #L = utils.graph_to_lap(data)
            #sketch
            #args.Lx = utils.graph_to_lap(g) #graph.graph_dist(args, plot=False)
            args.Lx = data
            args.m = len(args.Lx)
            args.n = math.ceil(args.m/args.compress_fac) 
            loss, P, Ly = graph.graph_dist(args, plot=False)
            ones = torch.ones(args.n, args.n).triu()
        
            if False: #True: #False:
                cur_labels = node_labels[i][P.argmax(0)]
                #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])
                #order node labels            
                L_ordering = Ly.diag().argsort(descending=True)
                cur_labels = cur_labels[L_ordering] ####
                labels.append(cur_labels)
            L = utils.canonicalize_mx(Ly)        

            #'''
            L = L[ones > 0]        
            #Ly_n.append( (L**2).sum())#.sqrt())
            #Ly_mx.append(L)
            cur_max_len = min(max_len, len(L))
            #Ly_mx[i][:cur_max_len] = L[:cur_max_len]
            data_sketch.append(L[:cur_max_len])
            #'''
        #torch.save(data_sketch, '{}_data_sketch.pt'.format(dataset_name))
    Ly_mx = torch.zeros(n_data, max_len)    
    for i, d_idx in enumerate(data_idx):
        cur_len = min(len(data_sketch[d_idx]), max_len)
        Ly_mx[i][:cur_len] = data_sketch[d_idx][:cur_len] #Ly_mx[:cur_max_len]
        
    #torch.save(Ly_mx, 'enzyme_data_lap.pt')
    #labels = torch.stack(labels)
    #data_t = torch.stack(Ly_mx) #.t()
    data_t = Ly_mx
    #data_t = torch.cat((data_t, labels*.03), -1)
    ##Ly_n = torch.Tensor(Ly_n)
    cur_ranks_ = torch.zeros(n_queries, args.k, dtype=torch.int64)
    
    for i, q_idx in enumerate(tqdm(query_idx, desc='queries')):
        '''
        args.Lx = q
        args.m = len(q)
        args.n = math.ceil(args.m/5) #args.lo_dim #
        loss, P, Lx = graph.graph_dist(args, plot=False)        
        ones = torch.ones(args.n, args.n).triu()
        #Lx_mx.append(args.Lx)
        Lx_n = (args.Lx**2).sum() #.sqrt()
        if False: #True: #False:
            #pdb.set_trace()
            cur_labels = q_labels[i][P.argmax(0)]
            #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])
            #order node labels
            #pdb.set_trace()
            L_ordering = Lx.diag().argsort(descending=True)
            q_label = cur_labels[L_ordering] ###
            
        Lx = utils.canonicalize_mx(Lx)[ones > 0]
        '''
        #cur_max_len = min(max_len, len(Lx))
        Lx_ = torch.zeros(max_len)
        Lx_[:len(data_sketch[q_idx])] = data_sketch[q_idx]
        ##Lx = torch.cat((Lx, q_label*.03), -1)
        #pdb.set_trace()
        #dist = Lx_n + Ly_n - 2*torch.mm(Lx.view(1, -1), data_t.t()).view(-1)
        #pdb.set_trace()
        dist = torch.abs(Lx_.view(1,-1) - data_t).sum(-1)
        
        '''
        dist = []
        heat_q = netlsd.heat(Lx.numpy())
        for j,d in enumerate(heat_l):
            #dist.append(torch.norm(Lx-d, 1) )
            dist.append(netlsd.compare(heat_q, heat_l[j]))
        dist = torch.Tensor(dist)
        '''
        
        cur_ranks_[i] = torch.topk(dist, args.k, largest=False)[1] #np.argpartition(dist, args.k)[:args.k]
        #cur_cls = dataset_cls[cur_ranks_[i]]
        #pdb.set_trace()
        
    if dataset_cls is not None:
        #pdb.set_trace()
        pred = dataset_cls[cur_ranks_[:, 0].numpy()]        
        acc1 = np.equal(pred, tgt_cls).sum()
        print('top 1 acc using just mx dist ! {}'.format(acc1/n_queries))
        if True:
            return acc1/n_queries, data_sketch #cur_ranks_
        # top-ten voting #
        dataset_cls = torch.from_numpy(dataset_cls)        
        ones = torch.ones(50)
        pred10 = np.zeros(n_queries)
        for i, q in enumerate(query_idx):
            #cur_ranks = dataset_cls_t[ot_cost_ranks[i]]
            ranked = torch.zeros(100) #n_cls*2
            cur_ranks = dataset_cls[cur_ranks_[i, :30]]
            ranked.scatter_add_(src=ones, index=cur_ranks, dim=-1)
            #pdb.set_trace()
            pred10[i] = torch.argmax(ranked).item()            
        acc10 = np.equal(pred, tgt_cls).sum()
        print('top 30 voting acc using just mx dist ! {}'.format(acc10/n_queries))        
        
    return cur_ranks_

def classify_netlsd_var_len(dataset, data_idx, query_idx, args, node_labels, q_labels, dataset_cls=None, tgt_cls=None, data_sketch=None):
    """
    Initial dataset filtering to filter out candidates for GW distance. 
    dataset: dataset of laplacians, not sketched yet
    """
    print('using netlsd var len')
    n_data = len(data_idx)
    n_queries = len(query_idx)
    max_len = max([len(l) for l in dataset])
    #max_len = (max_len//5+1)*max_len//5 //2
    #five-fold compression, put in args
    max_len = (math.ceil(max_len/args.compress_fac)+1)*math.ceil(max_len/args.compress_fac) //2
    Lx_mx = []
    data_graphs = []
    Ly_n = []
    dataset_cls1 = []
    labels = []
    heat_l = []
    if data_sketch is None:
        data_sketch = []    
        for i, data in enumerate(tqdm(dataset)):
            n_nodes = len(data)        
            #L = utils.graph_to_lap(data)
            #sketch
            #args.Lx = utils.graph_to_lap(g) #graph.graph_dist(args, plot=False)
            args.Lx = data
            args.m = len(args.Lx)
            args.n = math.ceil(args.m/args.compress_fac) 
            loss, P, Ly = graph.graph_dist(args, plot=False)
            ones = torch.ones(args.n, args.n).triu()
            
            if False: #True: #False:
                cur_labels = node_labels[i][P.argmax(0)]
                #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])
                #order node labels            
                L_ordering = Ly.diag().argsort(descending=True)
                cur_labels = cur_labels[L_ordering] ####
                labels.append(cur_labels)
            L = utils.canonicalize_mx(Ly)            
            #'''
            L = L[ones > 0]        
            #Ly_n.append( (L**2).sum())#.sqrt())
            #Ly_mx.append(L)
            cur_max_len = min(max_len, len(L))
            #Ly_mx[i][:cur_max_len] = L[:cur_max_len]
            data_sketch.append(L[:cur_max_len])
            #'''
        
    def reconstruct_mx(upper):
        #reconstruct full mx from upper triangular part
        n_nodes = int(-.5 + np.sqrt(.25 + 2*len(upper)))
        #
        mx = torch.zeros((n_nodes, n_nodes))
        ones = torch.ones((n_nodes, n_nodes))
        try:
            mx[ones.triu()>0] = upper
        except Exception:
            print('exception!')
            pdb.set_trace()    
        mx += mx.triu(diagonal=1).t()
        return mx
    Ly_mx = torch.zeros(n_data, max_len)    
    for i, d_idx in enumerate(data_idx):        
        #Ly_mx[i][:len(data_sketch[d_idx])] = data_sketch[d_idx] #Ly_mx[:cur_max_len]
        L = reconstruct_mx(data_sketch[d_idx])
        #pdb.set_trace()
        heat_l.append(netlsd.heat(L.numpy())) #reconstruct!  #try normalization!?
        
    #torch.save(Ly_mx, 'enzyme_data_lap.pt')
    #labels = torch.stack(labels)
    #data_t = torch.stack(Ly_mx) #.t()
    data_t = Ly_mx
    #data_t = torch.cat((data_t, labels*.03), -1)
    ##Ly_n = torch.Tensor(Ly_n)
    cur_ranks_ = torch.zeros(n_queries, args.k, dtype=torch.int64)
    all_dist = torch.zeros(n_queries, len(heat_l))
    for i, q_idx in enumerate(tqdm(query_idx, desc='queries')):
        '''
        args.Lx = q
        args.m = len(q)
        args.n = math.ceil(args.m/5) #args.lo_dim #
        loss, P, Lx = graph.graph_dist(args, plot=False)        
        ones = torch.ones(args.n, args.n).triu()        
        #Lx_mx.append(args.Lx)
        Lx_n = (args.Lx**2).sum() #.sqrt()
        if False: #True: #False:
            #pdb.set_trace()
            cur_labels = q_labels[i][P.argmax(0)]
            #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])
            #order node labels
            #pdb.set_trace()
            L_ordering = Lx.diag().argsort(descending=True)
            q_label = cur_labels[L_ordering] ###
            
        Lx = utils.canonicalize_mx(Lx)[ones > 0]
        '''
        #cur_max_len = min(max_len, len(Lx))
        #Lx_ = torch.zeros(max_len)
        #Lx_[:len(data_sketch[q_idx])] = data_sketch[q_idx]
        ##Lx = torch.cat((Lx, q_label*.03), -1)
        #pdb.set_trace()
        #dist = Lx_n + Ly_n - 2*torch.mm(Lx.view(1, -1), data_t.t()).view(-1)
        #pdb.set_trace()
        #dist = torch.abs(Lx_.view(1,-1) - data_t).sum(-1)        
        #'''
        dist = []
        L = reconstruct_mx(data_sketch[q_idx])
        heat_q = netlsd.heat(L.numpy()) ##reconstruct! 
        for j,d in enumerate(heat_l):
            #dist.append(torch.norm(Lx-d, 1) )
            dist.append(netlsd.compare(heat_q, heat_l[j]))
        dist = torch.Tensor(dist)
        #'''        
        #cur_ranks_[i] = torch.topk(dist, args.k, largest=False)[1] #np.argpartition(dist, args.k)[:args.k]
        cur_ranks_[i] = torch.topk(dist, args.k, largest=False)[1]
        all_dist[i] = dist #torch.topk(dist, len(dist), largest=False)[0]
        #cur_cls = dataset_cls[cur_ranks_[i]]
        #pdb.set_trace()

    print('top ranks ', cur_ranks_.squeeze())
    train_cls = np.array([dataset_cls[i] for i in data_idx])
    if dataset_cls is not None:
        #pdb.set_trace()
        pred = train_cls[cur_ranks_[:, 0].numpy()]        
        acc1 = np.equal(pred, tgt_cls).sum()
        print('top 1 acc using just mx dist ! {}'.format(acc1/n_queries))
        ###
        n_cls = 2
        pred_cls = []
        for i in range(len(all_dist)):
            best_cls, best_dist = -1, sys.maxsize
            for c in [0,1]: #range(n_cls):
                cur_dist = all_dist[i][dataset_cls==c].mean().item()
                if cur_dist < best_dist:
                    best_dist, best_cls = cur_dist, c
            pred_cls.append(best_cls)
        acc2 = np.equal(np.array(pred_cls), tgt_cls).sum()/n_queries
        print('AVG dist acc ', acc2)
        ###
        if True:
            return acc1/n_queries, data_sketch #cur_ranks_
        

def sketch_dataset_dim(dataset, node_labels, args, method):
    """
    Sketch dataset using the desired method to the desired cardinalities.
    Input: dataset given as graph Laplacians.
    Returns: compressed graph Laplacians.
    """
    Ly_l = []
    n = args.lo_dim
    if method == 'copt':
        for i, data in enumerate(tqdm(dataset, desc='sketching')):
            args.n = lo_dim
            args.n_epochs = 230 #250
            args.Lx = utils.graph_to_lap(g) #graph.graph_dist(args, plot=False)        
            args.m = len(args.Lx) 
            loss, P, Ly = graph.graph_dist(args, plot=False)            
        Ly_l.append(Ly)
        
    '''
    if s:
        m = len(data)        
        compress_ratio = args.lo_dim / m
        if method in ['heavy_edge', 'variational_neightborhood', 'variation_edges', 'affinity_GS', 'algebraic_JC']:
            C, Gc, Call, Gall = coarsen(g, K=k, r=r, method=method)
            compress_ratio = s
            args.n = math.ceil(args.m/args.compress_fac)
        elif method == 'otc':
            s
        elif method == 'copt':
    '''    
    return Ly_l

    
def sketch_dataset(dataset, node_labels, args, tgt_n=None):
    """
    Sketch a given dataset to desired compress factor.
    Input: dataset is given as as Laplacian matrices of graphs.
    """
    max_len = max([len(g) for g in dataset])    
    #max_len = (max_len//args.compress_fac+1)*max_len//args.compress_fac //2
    data_sketch = []
    P_sketch = []
    #all_node_labels
    min_label = min([min(l) for l in node_labels])
    max_label = max([max(l) for l in node_labels])
    pdb.set_trace()
    node_labels -= min_label ###
    n_labels = int(max_label - min_label) + 1
    #cls_labels_ = torch.arange(max_label-min_label).unsqueeze(-1)
    normalize_Ly = False # True Normalize during runtime for flexibility
    
    for i, data in enumerate(tqdm(dataset)):
        n_nodes = len(data)        
        #L = utils.graph_to_lap(data)
        #sketch
        #args.Lx = utils.graph_to_lap(g) #graph.graph_dist(args, plot=False)
        args.Lx = data
        args.m = len(args.Lx)
        args.n = math.ceil(args.m/args.compress_fac) if tgt_n is None else tgt_n
        #P has shape eg 19x7
        loss, P, Ly = graph.graph_dist(args, plot=False)
        if normalize_Ly:
            Ly /= n_nodes
            ###
            #Ly /= Ly.diag().sum()
            
        ones = torch.ones(args.n, args.n).triu()
        #compressed number of nodes
        n_comp = P.shape[1]
        if True: #True: #False:
            #pdb.set_trace()
            #cls_labels  = cls_labels_.repeat(1, n_nodes)
            prob_labels = torch.zeros(n_labels, n_comp )
            cur_labels = torch.from_numpy(node_labels[i]).to(torch.int64).unsqueeze(-1).expand(-1, n_comp)
            #temp = 1 #2
            #probs = torch.nn.functional.softmax(P/temp, dim=0)
            probs = P / P.sum(0, keepdim=True)
            try:
                prob_labels.scatter_add_(index=cur_labels, src=probs, dim=0)
            except RuntimeError as e:
                print('scatter add error!')
                pdb.set_trace()
            
            cur_labels = prob_labels.t() #node_labels[i][P.argmax(0)]
            ###cur_labels = node_labels[i][P.argmax(0)]
            
            #order node labels to be consistent with canonicalization
            L_ordering = Ly.diag().argsort(dim=0)
            cur_labels = cur_labels[L_ordering] ####
            #labels.append(cur_labels)
        L = utils.canonicalize_mx(Ly)        
        #pdb.set_trace()
        #'''
        L = L[ones > 0]        
        #Ly_n.append( (L**2).sum())#.sqrt())
        #Ly_mx.append(L)
        #cur_max_len = min(max_len, len(L))        
        ##data_sketch.append(L[:cur_max_len])
        data_sketch.append(L)
        P_sketch.append(cur_labels)
        #'''
    #torch.save(data_sketch, '{}_data_sketch.pt'.format(dataset_name))
    return data_sketch, P_sketch

def classify_mutag_l1(dataset, queries, args, node_labels, q_labels, dataset_cls=None, tgt_cls=None):#dataset_cls, target, args):
    """
    Can be used for classifying real world graphs eg enzymes or mutag dataset.
    dataset: laplacians of sketched graphs.
    """
    #with open(args.graph_fname, 'rb') as f:
    #    graphs = pickle.read(f)
    n_data = len(dataset)
    n_queries = len(queries)
    #ot_cost = np.zeros((len(queries), len(dataset)))
    gw_cost = np.zeros((len(queries), len(dataset)))
    #gwdist = Fused_Gromov_Wasserstein_distance(alpha=args.alpha,features_metric='sqeuclidean')
    Ly_mx = []
    Lx_mx = []
    data_graphs = []
    Ly_n = []
    args.lo_dim = 8
    ones = torch.ones(args.lo_dim, args.lo_dim).triu()
    dataset_cls1 = []
    labels = []
    heat_l = []
    for i, data in enumerate(tqdm(dataset)):
        n_nodes = len(data)
        #assert n_nodes >= 10
        
        #L = utils.graph_to_lap(data)
        #sketch
        #args.Lx = utils.graph_to_lap(g) #graph.graph_dist(args, plot=False)
        args.Lx = data
        args.m = len(args.Lx)
        args.n = args.lo_dim #8 #math.ceil(args.m/5)        
        loss, P, Ly = graph.graph_dist(args, plot=False)
        #pdb.set_trace()
        #pdb.set_trace()
        #lo_graphs.append(Ly)
        
        if False: #True: #False:
            cur_labels = node_labels[i][P.argmax(0)]
            #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])
            #order node labels
            
            L_ordering = Ly.diag().argsort(descending=True)
            cur_labels = cur_labels[L_ordering] ####
            labels.append(cur_labels)
        L = utils.canonicalize_mx(Ly)        
        
        L = L[ones > 0]        
        Ly_n.append( (L**2).sum())#.sqrt())
        Ly_mx.append(L)
        
        #heat_l.append(netlsd.heat(L.numpy()))
        #data_graphs.append(gwGraph.Graph(dataset0[i]))
        
    #torch.save(Ly_mx, 'enzyme_data_lap.pt')
    #labels = torch.stack(labels)
    data_t = torch.stack(Ly_mx) #.t()
    #data_t = torch.cat((data_t, labels*.03), -1)
    ##Ly_n = torch.Tensor(Ly_n)
    cur_ranks_ = torch.zeros(n_queries, args.k, dtype=torch.int64)
    #pdb.set_trace()
    
    for i, q in enumerate(tqdm(queries, desc='queries')):
        #Lx = utils.graph_to_lap(q)
        args.Lx = q
        args.m = len(q)
        args.n = args.lo_dim #math.ceil(args.m/5)
        loss, P, Lx = graph.graph_dist(args, plot=False)        
        
        #q_graph = gwGraph.Graph(queries0[i])        
        #Lx_mx.append(args.Lx)
        Lx_n = (args.Lx**2).sum() #.sqrt()
        if False: #True: #False:
            #pdb.set_trace()
            cur_labels = q_labels[i][P.argmax(0)]
            #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])
            #order node labels
            #pdb.set_trace()
            L_ordering = Lx.diag().argsort(descending=True)
            q_label = cur_labels[L_ordering] ###
            
        Lx = utils.canonicalize_mx(Lx)[ones > 0]
        #Lx = torch.cat((Lx, q_label*.03), -1)
        #pdb.set_trace()
        #dist = Lx_n + Ly_n - 2*torch.mm(Lx.view(1, -1), data_t.t()).view(-1)
        #pdb.set_trace()
        dist = torch.abs(Lx.view(1,-1) - data_t).sum(-1)
        
        '''
        dist = []
        heat_q = netlsd.heat(Lx.numpy())
        for j,d in enumerate(Ly_mx):
            #dist.append(torch.norm(Lx-d, 1) )
            dist.append(netlsd.compare(heat_q, heat_l[i]))
        dist = torch.Tensor(dist)
        '''
        cur_ranks_[i] = torch.topk(dist, args.k, largest=False)[1] #np.argpartition(dist, args.k)[:args.k]
        #cur_cls = dataset_cls[cur_ranks_[i]]
        #pdb.set_trace()
        
    if dataset_cls is not None:
        pdb.set_trace()
        pred = dataset_cls[cur_ranks_[:, 0].numpy()]        
        acc1 = np.equal(pred, tgt_cls).sum()
        print('top 1 acc using just mx dist ! {}'.format(acc1/n_queries))
        # top-ten voting #
        dataset_cls = torch.from_numpy(dataset_cls)        
        ones = torch.ones(50)
        pred10 = np.zeros(n_queries)
        for i, q in enumerate(queries):
            #cur_ranks = dataset_cls_t[ot_cost_ranks[i]]
            ranked = torch.zeros(100) #n_cls*2
            cur_ranks = dataset_cls[cur_ranks_[i, :30]]
            ranked.scatter_add_(src=ones, index=cur_ranks, dim=-1)
            #pdb.set_trace()
            pred10[i] = torch.argmax(ranked).item()            
        acc10 = np.equal(pred, tgt_cls).sum()
        print('top 30 voting acc using just mx dist ! {}'.format(acc10/n_queries))        
        
    return cur_ranks_

def classify_l1_var_len_(dataset, data_idx, query_idx, args, node_labels, q_labels, dataset_cls=None, tgt_cls=None, data_sketch=None):
    """
    Initial dataset filtering to filter out candidates for GW distance. 
    dataset: dataset of laplacians, not sketched yet
    """
    print('using sketching l1 var len')
    n_data = len(data_idx)
    n_queries = len(queries)
    max_len = max([len(l) for l in dataset])
    max_len = (max_len//args.compress_fac+1)*max_len//args.compress_fac //2
    
    Lx_mx = []
    data_graphs = []
    Ly_n = []
    dataset_cls1 = []
    labels = []
    
    for i, data in enumerate(tqdm(dataset)):
        n_nodes = len(data)        
        #L = utils.graph_to_lap(data)
        #sketch
        #args.Lx = utils.graph_to_lap(g) #graph.graph_dist(args, plot=False)
        args.Lx = data
        args.m = len(args.Lx)
        args.n = math.ceil(args.m/args.compress_fac) 
        loss, P, Ly = graph.graph_dist(args, plot=False)
        ones = torch.ones(args.n, args.n).triu()
        #pdb.set_trace()
        #pdb.set_trace()
        #lo_graphs.append(Ly)
        
        if False: #True: #False:
            cur_labels = node_labels[i][P.argmax(0)]
            #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])
            #order node labels            
            L_ordering = Ly.diag().argsort(descending=True)
            cur_labels = cur_labels[L_ordering] ####
            labels.append(cur_labels)
        L = utils.canonicalize_mx(Ly)        
        
        #'''
        L = L[ones > 0]        
        #Ly_n.append( (L**2).sum())#.sqrt())
        #Ly_mx.append(L)
        cur_max_len = min(max_len, len(L))
        Ly_mx[i][:cur_max_len] = L[:cur_max_len]
        #'''
        
    Ly_mx = torch.zeros(n_data, max_len)    
    for i, d_idx in enumerate(data_idx):
        
        Ly_mx[i][:len(data_sketch[d_idx])] = data_sketch[d_idx] #Ly_mx[:cur_max_len]
        
    #torch.save(Ly_mx, 'enzyme_data_lap.pt')
    #labels = torch.stack(labels)
    #data_t = torch.stack(Ly_mx) #.t()
    data_t = Ly_mx
    #data_t = torch.cat((data_t, labels*.03), -1)
    ##Ly_n = torch.Tensor(Ly_n)
    cur_ranks_ = torch.zeros(n_queries, args.k, dtype=torch.int64)
    #pdb.set_trace()
    
    for i, q_idx in enumerate(tqdm(query_idx, desc='queries')):
        
        args.Lx = q
        args.m = len(q)
        args.n = math.ceil(args.m/args.compress_fac) #args.lo_dim #
        loss, P, Lx = graph.graph_dist(args, plot=False)        
        ones = torch.ones(args.n, args.n).triu()
        #q_graph = gwGraph.Graph(queries0[i])        
        #Lx_mx.append(args.Lx)
        Lx_n = (args.Lx**2).sum() #.sqrt()
        if False: #True: #False:
            #pdb.set_trace()
            cur_labels = q_labels[i][P.argmax(0)]
            #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])
            #order node labels
            #pdb.set_trace()
            L_ordering = Lx.diag().argsort(descending=True)
            q_label = cur_labels[L_ordering] ###
            
        Lx = utils.canonicalize_mx(Lx)[ones > 0]
        
        cur_max_len = min(max_len, len(Lx))
        Lx_ = torch.zeros(max_len)
        Lx_[:cur_max_len] = Lx[:cur_max_len]
        ##Lx = torch.cat((Lx, q_label*.03), -1)
        #pdb.set_trace()
        #dist = Lx_n + Ly_n - 2*torch.mm(Lx.view(1, -1), data_t.t()).view(-1)
        #pdb.set_trace()
        dist = torch.abs(Lx_.view(1,-1) - data_t).sum(-1)
        
        '''
        dist = []
        heat_q = netlsd.heat(Lx.numpy())
        for j,d in enumerate(heat_l):
            #dist.append(torch.norm(Lx-d, 1) )
            dist.append(netlsd.compare(heat_q, heat_l[j]))
        dist = torch.Tensor(dist)
        '''
        
        cur_ranks_[i] = torch.topk(dist, args.k, largest=False)[1] #np.argpartition(dist, args.k)[:args.k]
        #cur_cls = dataset_cls[cur_ranks_[i]]
        #pdb.set_trace()
        
    if dataset_cls is not None:
        pdb.set_trace()
        pred = dataset_cls[cur_ranks_[:, 0].numpy()]        
        acc1 = np.equal(pred, tgt_cls).sum()
        print('top 1 acc using just mx dist ! {}'.format(acc1/n_queries))
        if True:
            return acc1 #cur_ranks_
        
#cur_idx = classify_filter
def classify_filter(dataset, target, args, train_idx, query_idx): #(dataset, queries, args, dataset_cls=None, tgt_cls=None):#dataset_cls, target, args):
    """
    Initial dataset filtering to filter out candidates for GW distance. 
    """
    #args.alpha = 0.8
    #with open(args.graph_fname, 'rb') as f:
    #    graphs = pickle.read(f)
    n_data = len(dataset)
    n_queries = len(query_idx)
    #ot_cost = np.zeros((len(queries), len(dataset)))
    #gw_cost = np.zeros((len(queries), len(dataset)))
    #gwdist = Fused_Gromov_Wasserstein_distance(alpha=args.alpha,features_metric='sqeuclidean')
    Ly_mx = []
    Lx_mx = []
    data_graphs = []
    Ly_n = []
    ones = torch.ones(len(dataset[0]), len(dataset[0])) #args.lo_dim, args.lo_dim).triu()
    for i, data_idx in enumerate(train_idx):
        data = dataset[data_idx]
        #n_nodes = len(data.nodes())
        #L = utils.graph_to_lap(data)
        #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])
        L = utils.canonicalize_mx(data)
        L = L[ones > 0]  
        Ly_n.append( (L**2).sum())#.sqrt())
        Ly_mx.append(L)
        #data_graphs.append(gwGraph.Graph(dataset0[i]))
        
    data_t = torch.stack(Ly_mx) #.t()
    Ly_n = torch.Tensor(Ly_n)
    cur_ranks_ = torch.zeros(n_queries, args.k, dtype=torch.int64)
    #pdb.set_trace()
    
    for i, q_idx in enumerate(tqdm(query_idx, desc='queries')):
        Lx = dataset[q_idx]
        #Lx = utils.graph_to_lap(q)
        #pdb.set_trace()
        Lx = utils.canonicalize_mx(Lx)
        args.Lx = Lx[ones>0]
        #q_graph = gwGraph.Graph(queries0[i])
        args.m = len(Lx)
        Lx_mx.append(args.Lx)
        Lx_n = (args.Lx**2).sum() #.sqrt()
        Lx = args.Lx
        #pdb.set_trace()
        #use either l1 or l2 distance for retrieval.
        #dist = Lx_n + Ly_n - 2*torch.mm(Lx.view(1, -1), data_t.t()).view(-1)
        dist = torch.abs(Lx.view(1,-1) - data_t).sum(-1)
        #dist = []
        #for j,d in enumerate(Ly_mx):
        #    dist.append(torch.norm(Lx-d,2) )
        #dist = torch.Tensor(dist)
        #
        cur_ranks_[i] = torch.topk(dist, args.k, largest=False)[1] #np.argpartition(dist, args.k)[:args.k]
        #cur_cls = dataset_cls[cur_ranks_[i]]
        
    
    train_cls = np.array([target[int(i)] for i in train_idx]) #target[train_idx]
    if True: #dataset_cls is not None:
        tgt_cls = np.array([target[int(i)] for i in query_idx])
        
        pred = target[cur_ranks_[:, 0].numpy()]        
        acc1 = np.equal(pred, tgt_cls).sum()/n_queries
        print('top 1 acc using just mx dist ! {}'.format(acc1))
        # top-ten voting #
        '''
        dataset_cls = torch.from_numpy(dataset_cls)        
        ones = torch.ones(50)
        pred10 = np.zeros(n_queries)
        for i, q in enumerate(queries):
            #cur_ranks = dataset_cls_t[ot_cost_ranks[i]]
            ranked = torch.zeros(100) #n_cls*2
            cur_ranks = dataset_cls[cur_ranks_[i, :30]]
            ranked.scatter_add_(src=ones, index=cur_ranks, dim=-1)
            #pdb.set_trace()
            pred10[i] = torch.argmax(ranked).item()        
        acc10 = np.equal(pred, tgt_cls).sum()
        print('top 10 voting acc using just mx dist ! {}'.format(acc10/n_queries))        
        '''
    return cur_ranks_, acc1

def classify_filter_copt(dataset, queries, args, dataset_cls=None, tgt_cls=None):#dataset_cls, target, args):
    """
    Initial dataset filtering to filter out candidates for GW distance. 
    """
    #args.alpha = 0.8
    #with open(args.graph_fname, 'rb') as f:
    #    graphs = pickle.read(f)
    n_data = len(dataset)
    n_queries = len(queries)
    #ot_cost = np.zeros((len(queries), len(dataset)))
    gw_cost = np.zeros((len(queries), len(dataset)))
    #gwdist = Fused_Gromov_Wasserstein_distance(alpha=args.alpha,features_metric='sqeuclidean')
    Ly_mx = []
    Lx_mx = []
    data_graphs = []
    Ly_n = []

    for i, data in enumerate(dataset):
        #n_nodes = len(data.nodes())
        #L = utils.graph_to_lap(data)
        #pdb.set_trace()
        L = reconstruct_mx(data)
        ones = torch.ones(len(L), len(L)).triu()
        #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])
        L = utils.canonicalize_mx(L)
        L = L[ones > 0]   
        Ly_n.append( (L**2).sum())#.sqrt())
        Ly_mx.append(L)
        #data_graphs.append(gwGraph.Graph(dataset0[i]))
        
    data_t = torch.stack(Ly_mx) #.t()
    Ly_n = torch.Tensor(Ly_n)
    cur_ranks_ = torch.zeros(n_queries, args.k, dtype=torch.int64)
    #pdb.set_trace()
    
    for i, q in enumerate(tqdm(queries, desc='queries')):
        #Lx = utils.graph_to_lap(q)
        #Lx = utils.canonicalize_mx(Lx)        
        Lx = reconstruct_mx(q)
        ones = torch.ones(len(Lx), len(Lx)).triu()
        args.Lx = Lx[ones>0]
        #q_graph = gwGraph.Graph(queries0[i])
        
        Lx_mx.append(args.Lx)
        Lx_n = (args.Lx**2).sum() #.sqrt()
        Lx = args.Lx
        #pdb.set_trace()
        #use either l1 or l2 distance for retrieval.
        #dist = Lx_n + Ly_n - 2*torch.mm(Lx.view(1, -1), data_t.t()).view(-1)
        dist = torch.abs(Lx.view(1,-1) - data_t).sum(-1)
        #dist = []
        #for j,d in enumerate(Ly_mx):
        #    dist.append(torch.norm(Lx-d,2) )
        #dist = torch.Tensor(dist)
        #
        cur_ranks_[i] = torch.topk(dist, args.k, largest=False)[1] #np.argpartition(dist, args.k)[:args.k]
        #cur_cls = dataset_cls[cur_ranks_[i]]
        #pdb.set_trace()
        
    if dataset_cls is not None:
        
        pred = dataset_cls[cur_ranks_[:, 0].numpy()]        
        acc1 = np.equal(pred, tgt_cls).sum()/n_queries
        print('top 1 acc using just mx dist ! {}'.format(acc1))
        '''
        # top-ten voting #
        dataset_cls = torch.from_numpy(dataset_cls)
        
        ones = torch.ones(50)
        pred10 = np.zeros(n_queries)
        for i, q in enumerate(queries):
            #cur_ranks = dataset_cls_t[ot_cost_ranks[i]]
            ranked = torch.zeros(100) #n_cls*2
            cur_ranks = dataset_cls[cur_ranks_[i, :30]]
            ranked.scatter_add_(src=ones, index=cur_ranks, dim=-1)
            #pdb.set_trace()
            pred10[i] = torch.argmax(ranked).item()        
        acc10 = np.equal(pred, tgt_cls).sum()
        print('top 10 voting acc using just mx dist ! {}'.format(acc10/n_queries))        
        '''
    return cur_ranks_, acc1

def classify_filter_svd(dataset, queries, args, dataset_cls=None, tgt_cls=None):#dataset_cls, target, args):
    """
    Initial dataset filtering for GW distance using svd.
    """
    #args.alpha = 0.8
    #with open(args.graph_fname, 'rb') as f:
    #    graphs = pickle.read(f)
    n_data = len(dataset)
    n_queries = len(queries)
    #ot_cost = np.zeros((len(queries), len(dataset)))
    gw_cost = np.zeros((len(queries), len(dataset)))
    #gwdist = Fused_Gromov_Wasserstein_distance(alpha=args.alpha,features_metric='sqeuclidean')
    Ly_mx = []
    Lx_mx = []
    data_graphs = []
    Ly_n = []
    eval_topk = 3 #2
    max_n_nodes = len(dataset[0].nodes())
    data_t = torch.zeros(n_data, eval_topk*max_n_nodes) #torch.stack(Ly_mx).t()
    ones = torch.ones(args.lo_dim, args.lo_dim).triu()
    for i, data in enumerate(dataset):
        n_nodes = len(data.nodes())
        L = utils.graph_to_lap(data)
        #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])
        L = graph.mx_svd(L, topk=eval_topk).view(-1)
        #pdb.set_trace()
        #L = L[ones > 0]
        Ly_n.append( (L**2).sum())#.sqrt())
        #Ly_mx.append(L)
        data_t[i, :len(L)] = L
        #data_graphs.append(gwGraph.Graph(dataset0[i]))
        
    #data_t = torch.stack(Ly_mx).t()
    data_t = data_t.t()
    Ly_n = torch.Tensor(Ly_n)
    cur_ranks_ = torch.zeros(n_queries, args.k, dtype=torch.int64)
    #pdb.set_trace()
    
    for i, q in enumerate(tqdm(queries, desc='queries')):
        Lx = utils.graph_to_lap(q)
        args.Lx = graph.mx_svd(Lx, topk=eval_topk).view(-1) #Lx[ones>0]
        #q_graph = gwGraph.Graph(queries0[i])
        args.m = len(q.nodes())        
        Lx_mx.append(args.Lx)
        Lx_n = (args.Lx**2).sum() #.sqrt()
        Lx = args.Lx
        if len(Lx) < max_n_nodes*eval_topk:
            zeros = torch.zeros(max_n_nodes*eval_topk)
            zeros[:len(Lx)] = Lx
            Lx = zeros
        #pdb.set_trace()
        try:
            dist = Lx_n + Ly_n - 2*torch.mm(Lx.view(1, -1), data_t).view(-1)
            #dist = torch.abs(Lx.view(1,-1) - data_t.t()).sum(-1)
        except RuntimeError:
            pdb.set_trace()
        #dist = []
        #for j,d in enumerate(Ly_mx):
        #    dist.append(torch.norm(Lx-d,2) )
        #dist = torch.Tensor(dist)
        
        cur_ranks_[i] = torch.topk(dist, args.k, largest=False)[1] #np.argpartition(dist, args.k)[:args.k]
        #cur_cls = dataset_cls[cur_ranks_[i]]
        #pdb.set_trace()
    #pdb.set_trace()
    if dataset_cls is not None:
        pred = dataset_cls[cur_ranks_[:, 0].numpy()]        
        acc1 = np.equal(pred, tgt_cls).sum()
        print('top 1 acc using just SVD dist ! {}'.format(acc1/n_queries))
        # top-ten voting #
        dataset_cls = torch.from_numpy(dataset_cls)
        
        ones = torch.ones(50)
        pred10 = np.zeros(n_queries)
        for i, q in enumerate(queries):
            #cur_ranks = dataset_cls_t[ot_cost_ranks[i]]
            ranked = torch.zeros(100) #n_cls*2
            cur_ranks = dataset_cls[cur_ranks_[i, :30]]
            ranked.scatter_add_(src=ones, index=cur_ranks, dim=-1)
            #pdb.set_trace()
            pred10[i] = torch.argmax(ranked).item()        
        acc10 = np.equal(pred, tgt_cls).sum()
        print('top 10 voting acc using just mx dist ! {}'.format(acc10/n_queries))        
    
    return cur_ranks_
    
#'''
def classify(dataset, queries, cur_idx, dataset_cls, target, args, dataset0=None, queries0=None):
    """
    Retrieve nearest neighbor and classify graphs, *after* results were filtered out, by eg spectral projections.
    dataset0, queries0 are original, non-sketched graphs.
    """
    args.alpha = 0.8
    if dataset0 is None:
        dataset0 = dataset
        queries0 = queries    
    n_data = len(dataset)
    n_queries = len(queries)
    ot_cost = np.zeros((len(queries), len(dataset)))
    gw_cost = np.zeros((len(queries), args.k))
    gwdist = Fused_Gromov_Wasserstein_distance(alpha=args.alpha,features_metric='sqeuclidean')
    Ly_mx = []
    Lx_mx = []
    data_graphs = []
    for i, data in enumerate(dataset):
        n_nodes = len(data.nodes())
        L = utils.graph_to_lap(data)
        #Ly_mx.append(L[torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1) > 0])
        Ly_mx.append(L)
        data_graphs.append(gwGraph.Graph(dataset0[i]))
    #pdb.set_trace()
    for i, q in enumerate(tqdm(queries, desc='queries')):
        Lx = utils.graph_to_lap(q)
        args.Lx = Lx
        q_graph = gwGraph.Graph(queries0[i])
        args.m = len(q.nodes())
        Lx_mx.append(args.Lx)        
        #for j, data in enumerate(dataset):
        #pdb.set_trace()
        for j in range(len(cur_idx[i])):
            data = dataset[cur_idx[i][j]]
            gw_cost[i][j] = gwdist.graph_d(q_graph, data_graphs[cur_idx[i][j] ])  
            
    gw_cost_ = torch.from_numpy(gw_cost)
    gw_cost_ranks = torch.argsort(gw_cost_, -1)[:, :min(args.n_per_cls, 1)]    
    ones = torch.ones(100)  #args.n_per_cls*2 (n_cls*2)
    #ot_cls = -np.ones(n_queries)
    gw_cls = -np.ones(n_queries)
    combine_cls = np.ones(n_queries)
    dataset_cls_t = torch.from_numpy(dataset_cls)
    
    for i in range(n_queries): #for each cls
        ranks_i = cur_idx[i][ gw_cost_ranks[i]]
        #cur_ranks = dataset_cls_t[gw_cost_ranks[i]]
        cur_ranks = dataset_cls_t[ranks_i]
        ranked = torch.zeros(100) #n_cls*2)
        ranked.scatter_add_(src=ones, index=cur_ranks, dim=-1)
        gw_cls[i] = torch.argmax(ranked).item()
        ## combined
        #ranked = torch.zeros(100) #n_cls*2)
        #ranked.scatter_add_(src=ones, index=cur_ranks_ot, dim=-1).scatter_add_(src=ones, index=cur_ranks, dim=-1)
        #combine_cls[i] = torch.argmax(ranked).item()
    
    gw_acc = np.equal(gw_cls, target).sum() / len(target)
    print('gw acc ', gw_acc)
    return


def sketch_graph(graphs, lo_dim, args):
    '''
    Run graph sketching with given method.
    Input: graphs: graphs to be dimension-reduced for. In networkx Graphs format.
    '''
    args.n = lo_dim
    lo_graphs = []
    args.n_epochs = 230 #250
    for g in tqdm(graphs, desc='sketching'):        
        args.Lx = utils.graph_to_lap(g) #graph.graph_dist(args, plot=False)        
        args.m = len(args.Lx)        
        loss, P, Ly = graph.graph_dist(args, plot=False)        
        lo_graphs.append(utils.lap_to_graph(Ly))
        
    return lo_graphs

def test_FGW(args):
    """
    Test fused Gromov-Wasserstein distance
    """
    args.m = 8
    args.n = 2
    if args.fix_seed:
        torch.manual_seed(0)
    #args.Lx = torch.randn(args.m*(args.m-1)//2)  #torch.FloatTensor([[1, -1], [-1, 2]])
    #args.Lx = realize_upper(args.Lx, args.m)
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
        q_dim, data_dim = 50, 50 #100
        #q_dim, data_dim = 20, 30 #100
    else:
        q_dim, data_dim = 20, 10
        
    args.q_dim, args.data_dim = q_dim, data_dim
    args.n_per_cls = 130 #100 #20 #15 #5 #15 # 20 #5
    #create_graphs(30, args, 'data/graphs{}.pkl'.format(30), n_graphs=args.n_per_cls) #do 30 to 10
    if args.fix_seed or True: #temporary
        np.random.seed(0)
    if False: #False: #True:#False: #True: #False: #True: #False: #False:#True:#True: #False: #False: #True: #False:   
        #create_graphs(q_dim, args, 'data/queries{}rand.pkl'.format(q_dim), n_graphs=10)
        create_graphs(data_dim, args, 'data/train_graphs{}rand.pkl'.format(data_dim), n_graphs=args.n_per_cls) #do 30 to 10
        
    args.lo_dim = 15 #15 #q_dim
    lo_dim = args.lo_dim
    args.compress_fac = args.compress_fac if args.compress_fac > 0  else 2 #4 #2 #2 #4 #2 #2 #4
    if False:#True: #True:  #False: #True: #False: #True: #True: #False: #False: #True:
        #'''
        dataset, dataset_cls = utils.load_data('data/train_graphs{}rand.pkl'.format(data_dim))
        #dataset = dataset[10:-50]
        #dataset_cls = dataset_cls[10:-50]
        #dataset = dataset[:20]
        #dataset_cls = dataset_cls[:20]        
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
    args.sketch_dual = False
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
        
        #val_set, val_cls = utils.load_data('data/queries_sketch{}_{}rand.pkl'.format(data_dim, lo_dim))
    elif args.sketch_dual:
        #dataset_lo, dataset_cls_lo = utils.load_data('data/train_graphs_sketch{}_{}rand.pkl'.format(data_dim, lo_dim))
        #dataset0, dataset_cls0 = utils.load_data('data/train_graphs{}rand.pkl'.format(data_dim))
        
        #dataset0, dataset_lo, dataset_cls0 = create_and_sketch(data_dim, lo_dim, args, 'data/train_graphs{}dual{}_{}Feb1.pkl'.format(data_dim, lo_dim, args.n_per_cls), n_graphs=args.n_per_cls) #do 30 to 10
        dataset0, dataset_lo, dataset_cls0 = create_and_sketch(data_dim, lo_dim, args, 'data/train_graphs{}dual{}_{}.pkl'.format(data_dim, lo_dim, args.n_per_cls), n_graphs=args.n_per_cls) #do 30 to 10
        dataset_cls_lo = dataset_cls0
        
        #pdb.set_trace()
        #dataset_lo, dataset_cls_lo = s
        #np.random.seed(0)
        idx = np.random.permutation(len(dataset0))#.tolist()
        dataset1, dataset_cls1 = [dataset0[i] for i in idx], [dataset_cls0[i] for i in idx]
        dataset_lo1, dataset_cls_lo1 = [dataset_lo[i] for i in idx], [dataset_cls_lo[i] for i in idx]
        n_val = 180 #20
        n_data = 600 #200 
        dataset, dataset_cls = dataset1[n_val:n_val+n_data], dataset_cls1[n_val:n_val+n_data]
        val_set, val_cls = dataset1[:n_val], dataset_cls1[:n_val]
        dataset_lo, dataset_cls_lo = dataset_lo1[n_val:n_val+n_data], dataset_cls_lo1[n_val:n_val+n_data]
        val_set_lo, val_cls_lo = dataset_lo1[:n_val], dataset_cls_lo1[:n_val]
        
        #val_set, val_cls = utils.load_data('data/queries{}rand.pkl'.format(data_dim))
    #'''
    if args.sketch or args.sketch_dual:
        cls2label = {0:0, 1:1, 4:2, 5:3, 6:4, 9:5}
        labels = []
        dataset_cls = np.array([cls2label[c] for c in dataset_cls])
        val_cls = np.array([cls2label[c] for c in val_cls])
    #'''
    
    #dataset_cls is list
    #train_graph(dataset, dataset_cls, val_set, val_cls, args)
    #test retrieval using various methods
    test_retrieval = True #False    
    if test_retrieval:
        args.n_epochs = 350
        args.k = 1
        #method = 'variation_neighborhoods' #'copt' var neigh produces var neigh
        method = 'algebraic_JC'
        #BZR_MD7_otc.pkl
        method = 'otc'
        method = 'copt'
        #enz_data = torch.load('enzymes_lap.pt')
        #dataset_name = 'enzymes' #enzymes or mutag
        dataset_name = 'ENZYMES' #
        #dataset_name = 'PROTEINS'
        dataset_name = 'MSRC_9'
        dataset_name = 'BZR_MD' 

        method2tgt_n = {'PROTEINS': 13, 'BZR_MD': 7}
        if method in ['otc', 'variation_neighborhoods', 'heavy_edge', 'algebraic_JC']:
            tgt_n = method2tgt_n[dataset_name]
            path = 'data/'+dataset_name+'{}_{}.pkl'.format(tgt_n, method)  #. Has keys 'lap' and 'labels'
            with open(path, 'rb') as f:
                data = pickle.load(f)
                #data = torch.load(path)
            data_sketch, target = data['lap'], data['labels']
            target = np.array(target)
            shapes = []
            for d in data_sketch:
                if d.size(0) != tgt_n:
                    shapes.append(d.size())
            sizes = [d.shape for d in data_sketch]
            
        else:
            enz_data = torch.load('data/{}_lap.pt'.format(dataset_name))
            print('dataset: {} compress fac {}'.format( dataset_name, args.compress_fac))

            dataset = enz_data['lap']
            if dataset_name in ['enzymes', 'mutag']:
                #target = np.array([int(i) for i in utils.read_lines('mutagLabels.txt')])
                target = np.array([int(i) for i in utils.read_lines('{}Labels.txt'.format(dataset_name))])
                node_labels = enz_data['labels']
            else:
                target = np.array(enz_data['target'])
                node_labels = enz_data['labels']

            tgt_n = method2tgt_n[dataset_name]
            sketch_path = '{}_data_sketch{}tgt_{}.pt'.format(dataset_name, tgt_n, args.hike_interval)
            if os.path.exists(sketch_path):
                data_sketch = torch.load(sketch_path)
                node_labels, data_sketch = data_sketch['sketch_P'], data_sketch['data_sketch']
            else:
                data_sketch, sketch_P = sketch_dataset(dataset, node_labels, args, tgt_n=tgt_n)
                node_labels = sketch_P
                torch.save({'data_sketch':data_sketch, 'sketch_P':sketch_P, 'compress_fac':args.compress_fac} , sketch_path)
                #data_sketch = None
        
        if False: #True:
            kk = 2 #if dataset_name not in ['PROTEINS'] else 4 #5 #20 #20 #80 #80 #4
            #dataset = dataset[::kk]
            target = target[::kk]
            #node_labels = node_labels[::kk]
            data_sketch = data_sketch[::kk]
            print('total data set size ', len(dataset))
     
        #this is used for testing l2-based retrieval
        cur_k = args.k
        print('args.k! ', cur_k)
        args.k = cur_k #3 #3 #40 #24 #100 #@50 #50 #100
        n_runs = 20
        acc_ar = np.zeros(n_runs)
        for i in range(n_runs):
            if False:#True: #False: #True: #True: #False: #True: #True: #False: #True: #False: #True: #False: #True: #False: #True: #False: #True: #False: #True: #False: #True:#False: #True:#False: #True:
                cur_idx = classify_filter_svd(dataset, val_set, args, dataset_cls=dataset_cls, tgt_cls=val_cls)
            else:
                n_query = 100 #50
                n_data = len(data_sketch)
                query_idx = np.random.choice(n_data, size=n_query)
                all_idx = np.arange(n_data)
                ones = np.ones(n_data)
                ones[query_idx] = 0
                train_idx = all_idx[ones>0]                
                #query_idx = all_idx[:n_query] #dataset_idx[:n_query]
                #dataset_idx = all_idx[n_query:] #[n_query:]
                
                if method in ['copt']:
                    val_set_lo = [data_sketch[i] for i in query_idx]
                    val_cls = np.array([target[i] for i in query_idx ])
                    dataset = [data_sketch[i] for i in train_idx]
                    dataset_cls = np.array([target[i] for i in train_idx])
                    cur_idx, acc = classify_filter_copt(dataset, val_set_lo, args, dataset_cls=dataset_cls, tgt_cls=val_cls)  #dataset_cls=None, tgt_cls=None):#dataset_cls, target, args):
                else:
                    cur_idx, acc = classify_filter(data_sketch, target, args, train_idx, query_idx)
            acc_ar[i] = acc
        acc_mean = np.round(acc_ar.mean()*100, 4)
        acc_std = np.round(acc_ar.std()*100, 4)
        print('{} retrieval acc {}+-{} on {}'.format(method, acc_mean, acc_std, dataset_name))
        pdb.set_trace()
        #classify(dataset, val_set, cur_idx, dataset_cls, val_cls, args, dataset0=dataset, queries0=val_set)
        

    test_sketching = False #True
    #test_sketching = True
    if test_sketching:
        args.n_epochs = 350
        args.k = 1
        #enz_data = torch.load('enzymes_lap.pt')
        #dataset_name = 'enzymes' #enzymes or mutag
        dataset_name = 'ENZYMES' #
        #dataset_name = 'PROTEINS'
        dataset_name = 'MSRC_9'
        
        enz_data = torch.load('{}_lap.pt'.format(dataset_name))
        print('dataset: {} compress fac {}'.format( dataset_name, args.compress_fac))
        
        dataset = enz_data['lap']
        if dataset_name in ['enzymes', 'mutag']:
            #target = np.array([int(i) for i in utils.read_lines('mutagLabels.txt')])
            target = np.array([int(i) for i in utils.read_lines('{}Labels.txt'.format(dataset_name))])
            node_labels = enz_data['labels']
        else:
            target = np.array(enz_data['target'])
            node_labels = enz_data['labels']

        sketch_path = '{}_data_sketch{}_{}.pt'.format(dataset_name, args.compress_fac, args.hike_interval)
        if os.path.exists(sketch_path):
            data_sketch = torch.load(sketch_path)
            node_labels, data_sketch = data_sketch['sketch_P'], data_sketch['data_sketch']
        else:
            data_sketch, sketch_P = sketch_dataset(dataset, node_labels, args)
            node_labels = sketch_P
            torch.save({'data_sketch':data_sketch, 'sketch_P':sketch_P, 'compress_fac':args.compress_fac} , sketch_path)
            #data_sketch = None
        if True:
            kk = 1 #if dataset_name not in ['PROTEINS'] else 4 #5 #20 #20 #80 #80 #4
            dataset = dataset[::kk]
            target = target[::kk]
            node_labels = node_labels[::kk]
            data_sketch = data_sketch[::kk]
            print('total data set size ', len(dataset))
            #pdb.set_trace()
        sz_thresh = 8
        #idx!!!!
        target = np.array([target[i] for i in range(len(dataset)) if len(dataset[i])>=sz_thresh])
        node_labels = [node_labels[i] for i in range(len(dataset)) if len(dataset[i])>=sz_thresh]
        data_sketch = [data_sketch[i] for i in range(len(dataset)) if len(dataset[i])>=sz_thresh]
        #sketch_P = [sketch_P[i] for i in range(len(dataset)) if len(dataset[i])>=sz_thresh]
        dataset = [dataset[i] for i in range(len(dataset)) if len(dataset[i])>=sz_thresh]
        
        idx = np.random.permutation(len(dataset)) #[:10] #20
        
        if False:
            target = np.array([target[i] for i in idx])
            n_q = 50 #100
            queries = dataset[:n_q]
            q_labels = node_labels[:n_q]
            tgt_cls = target[:n_q]
            #n_q = 0
            dataset = dataset[n_q:]
            node_labels = node_labels[n_q:]
            dataset_cls = target[n_q:]        
            classify_l1_var_len(dataset, queries, args, node_labels, q_labels, dataset_cls=dataset_cls, tgt_cls=tgt_cls)
            #classify_graphs(dataset, queries, args, node_labels, q_labels, dataset_cls=dataset_cls, tgt_cls=tgt_cls)

        n_data = len(dataset)
        test_pct = .3 #3 #.33 #.5 # .33
        
        chunk_sz = math.ceil(n_data * test_pct)
        n_chunks = math.ceil(1/test_pct)
        n_chunks = 5
        print('percentage of data {} for test set! nchunks {}'.format(test_pct, n_chunks))
        acc_l = []
        args.dataset_name = dataset_name
        
        dataset = [dataset[i] for i in idx]
        node_labels = [node_labels[i] for i in idx]
        target = np.array([target[i] for i in idx])
        data_sketch = [data_sketch[i] for i in idx] 
        #sketch_P = [sketch_P[i] for i in idx]
        #pdb.set_trace()
        
        data_range = list(range(len(dataset))) # #np.arange(len(dataset)) # #list(range(len(dataset))) #
        random_chunk = True #False #True
        C_opt = None
        for i in range(n_chunks):
            if random_chunk:
                #pass
                query_idx = np.random.randint(0, high=len(dataset), size=(chunk_sz,), dtype=np.int64)
                data_idx = np.arange(len(dataset))
                data_idx[query_idx] = -1
                data_idx = data_idx[data_idx>-1]
                q_labels = None
                tgt_cls = target[query_idx]                
            else:
                low = i*chunk_sz
                high = min((i+1)*chunk_sz, n_data)                
                data_idx = data_range[:low] + data_range[high:]
                
                query_idx = data_range[low:high]  ##can do random!!
                #queries = dataset[low:high]
                q_labels = node_labels[low:high]
                tgt_cls = np.array(target[low:high])
            
            ##dataset_ = dataset[:low] + dataset[high:]
            ##node_labels_ = node_labels[:low] + node_labels[high:]
            ###dataset_cls = np.array(target[:low] + target[high:])
            dataset_cls = target
            if dataset_name in ['IMDB-BINARY']:
                acc, data_sketch = classify_netlsd_var_len(dataset, data_idx, query_idx, args, node_labels, q_labels, dataset_cls=dataset_cls, tgt_cls=tgt_cls, data_sketch=data_sketch)
            else:
                acc, data_sketch, C_opt = classify_svm_var_len(dataset, data_idx, query_idx, args, node_labels, q_labels, dataset_cls=dataset_cls, tgt_cls=tgt_cls, data_sketch=data_sketch, C_opt=C_opt)
                C_opt = None #reset each time for new grid search
            #acc, data_sketch = classify_l1_var_len(dataset, data_idx, query_idx, args, node_labels_, q_labels, dataset_cls=dataset_cls, tgt_cls=tgt_cls, data_sketch=data_sketch)
            do_classify_graphs = False
            if do_classify_graphs:
                classify_graphs(dataset_, queries, args, node_labels_, q_labels, dataset_cls=dataset_cls, tgt_cls=tgt_cls)
            acc_l.append(acc)
            print(acc)
            
        #torch.save(data_sketch, '{}_data_sketch.pt'.format(dataset_name))        
        print('mean acc: {}+-{}'.format( np.round(np.array(acc_l).mean(), 5), np.std(np.array(acc_l))))
        pdb.set_trace()

