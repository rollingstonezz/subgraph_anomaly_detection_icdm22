import os.path as osp
import matplotlib.pyplot as plt
import torch
import torch_geometric.transforms as T
import numpy as np
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU
import argparse
import os, sys
from tqdm import tqdm
import time
import pickle as pkl
import copy
import networkx as nx
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.utils import subgraph 
from torch.autograd import Variable
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter
from networkx.algorithms.components import strongly_connected_components
from torch_geometric.transforms import NormalizeFeatures
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools
from model.encoder import *
from model.decoder import *
from model.model import *
from model.smgnn import *

def add_anchors(data, args):
    # sample anchors 
    num_anchors = args.num_anchors
    G = to_networkx(data)
    anchors = np.random.choice(data.num_nodes, num_anchors, replace=False)
    dist_to_anchors = []
    for anchor in anchors:
        length = nx.single_source_shortest_path_length(G, anchor, cutoff=data.num_nodes//num_anchors)
        novalue_nodes = set([i for i in range(data.num_nodes)]) - set(length.keys())
        for each_node in novalue_nodes:
            length[each_node] = data.num_nodes//num_anchors+1
        dist_to_anchors.append(list(dict(sorted(length.items())).values()))
    dist_to_anchors = np.array(dist_to_anchors)

    # concat original x and distance to anchors
    data.x = torch.cat([data.x, torch.tensor(dist_to_anchors, dtype=torch.float).T], dim=-1)
    return data

def get_real_adj(data):
    # get real adj matrix
    num_neg = 1
    edge_index = data.edge_index
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index)
    real_adj_sparse = SparseTensor(row=edge_index[0], col=edge_index[1],
                        value=torch.ones(len(edge_index[0]), device=edge_index.device),
                        sparse_sizes=(data.num_nodes, data.num_nodes))
    real_adj = real_adj_sparse.to_dense()
    return real_adj

# load datasets
def load_structure_synthetic(args):
    n = args.size
    n_anomaly = int(n*args.anomaly_ratio)

    DATA_DIR = args.data_dir
    dataset = args.dataset #'small_world', 'random', 'complete', 'scale_free'
    anomaly_type = args.anomaly_type #'random', 'chain', 'complete'
    size = args.size # 1000 2000 
    FILE_NAME = os.path.join(DATA_DIR, 'structure_anomaly', dataset, anomaly_type, "size_{}.pkl".format(size))
    with open(FILE_NAME, 'rb') as file:
        data = pkl.load(file)

    anomaly_flag = np.array([False]*(data.num_nodes-n_anomaly) + [True]*n_anomaly)
    
    return data, anomaly_flag

def load_attribute_synthetic(args):

    DATA_DIR = args.data_dir
    FILE_NAME = "{}_{}_{}_{}.pkl".format(args.size, args.dim, args.anomaly_attr_ratio, args.diff_ratio)
    FILE_PATH = os.path.join(DATA_DIR, 'attribute_anomaly', FILE_NAME)
    
    with open(FILE_PATH, 'rb') as file:
        data = pkl.load(file)

    anomaly_flag = data.anomaly_flag.numpy()
    
    return data, anomaly_flag
def load_material(args):

    DATA_DIR = args.data_dir
    FILE_NAME = "{}.pkl".format(args.half_num)
    FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)
    
    with open(FILE_PATH, 'rb') as file:
        data = pkl.load(file)

    anomaly_flag = data.anomaly_flag.numpy()
    
    return data, anomaly_flag
def load_real_world(args):

    DATA_DIR = args.data_dir
    FILE_NAME = "{}.pkl".format(args.real_world_name)
    FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)
    
    with open(FILE_PATH, 'rb') as file:
        data = pkl.load(file)

    anomaly_flag = data.anomaly_flag.numpy()
    
    return data, anomaly_flag

def train(model, epoch, data):
    model.train()
    optimizer.zero_grad()
    x = data.x
    edge_index = data.edge_index
    z = model.encode(x, edge_index)
    pos_loss, neg_loss, attr_loss = model.recon_loss(z, edge_index, x=x)
    return pos_loss, neg_loss, attr_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_flag', type=str, default='structure_anomaly')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--real_world_name', type=str, default='email')
    parser.add_argument('--dataset', type=str, default='random')
    parser.add_argument('--anomaly_type', type=str, default='chain')
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--anomaly_ratio', type=float, default=0.02)
    parser.add_argument('--dim', type=int, default=50)
    parser.add_argument('--anomaly_scale', type=float, default=0.3)
    parser.add_argument('--anomaly_attr_ratio', type=float, default=0.2)
    parser.add_argument('--diff_ratio', type=int, default=5)
    parser.add_argument('--half_num', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--num_anchors', type=int, default=100)
    parser.add_argument('--embedding_channels', type=int, default=64)
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=90)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--beta_1', type=float, default=1.)
    parser.add_argument('--beta_2', type=float, default=1.)
    parser.add_argument('--q', type=int, default=90)
    parser.add_argument('--convergence', type=float, default=1e-4)
    parser.add_argument('--ending_rounds', type=int, default=1)
    args = parser.parse_args()

    # load data
    if args.data_flag == 'structure_anomaly':
        # load data
        data, anomaly_flag = load_structure_synthetic(args)
        RESULT_DIR = os.path.join(args.results_dir, 'structure_synthetic', args.dataset, args.anomaly_type)
        RESULT_FILE_NAME = "{}_{}_{}_{}_{}.txt".format(args.dataset, args.anomaly_type, args.size, args.embedding_channels, args.num_anchors)
    elif args.data_flag == 'attribute_anomaly':
        # load data
        data, anomaly_flag = load_attribute_synthetic(args)
        # set results file dir
        RESULT_DIR = os.path.join(args.results_dir, str(args.dim), str(args.anomaly_attr_ratio), str(args.diff_ratio))
        RESULT_FILE_NAME = "{}_{}_{}_{}_{}_{}.txt".format(str(args.dim), str(args.anomaly_attr_ratio), str(args.diff_ratio), args.size, args.embedding_channels, args.num_anchors)
    elif args.data_flag == 'material':
        # load data
        data, anomaly_flag = load_material(args)
        RESULT_DIR = os.path.join(args.results_dir, str(args.half_num))
        RESULT_FILE_NAME = "{}_{}_{}_{}.txt".format(str(args.half_num),args.size, args.embedding_channels, args.num_anchors)
    elif args.data_flag == 'real_world':
        # load data
        data, anomaly_flag = load_real_world(args)
        RESULT_DIR = os.path.join(args.results_dir, str(args.real_world_name))
        RESULT_FILE_NAME = "{}_{}_{}.txt".format(args.real_world_name, args.embedding_channels, args.num_anchors)

    # set results file dir
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        
    f = open(os.path.join(RESULT_DIR, RESULT_FILE_NAME), 'a')

    # add anchors
    if args.num_anchors > 0:
        data = add_anchors(data, args)
    
    # get real adj matrix
    real_adj = get_real_adj(data)

    ################################################
    ################initalize model#################
    ################################################
    
    # parameters
    out_channels = data.num_features
    num_features = data.num_features
    num_anchors = args.num_anchors
    epochs = args.epochs
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    embedding_channels = args.embedding_channels
    hidden_channels = args.hidden_channels

    # model
    num_layer = args.num_layers - 2
    innerproduct_decoder = InnerProductDecoder(embedding_channels, hidden_channels, out_channels, num_layers=2)
    #mlp_decoder = MLPDecoder(embedding_channels, hidden_channels, out_channels, num_layers=2)
    model = GAE(
        #encoder=EdgeConvEncoder(num_features, hidden_channels, embedding_channels, num_layers=num_layer, aggr='add'),
        encoder=GCNEncoder(num_features, hidden_channels, embedding_channels, num_layers=num_layer, aggr='add'),
        decoder=innerproduct_decoder#mlp_decoder,
    )
    # move to GPU (if available)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    # inizialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer_sp = torch.optim.Adam(model.parameters(), lr=.1)

    x = data.x
    edge_index = data.edge_index
    y = real_adj.cpu().numpy()
    
    prev_loss = 1e12
    flag = True
    accumulated_rounds = 1
    while flag:
        print('Iteration #%2d'%(accumulated_rounds))
        for epoch in range(1, args.epochs + 1):
            pos_loss, neg_loss, attr_loss = train(model, epoch, data)
            loss = beta_1*(pos_loss + neg_loss) + beta_2*attr_loss.mean()
            loss.backward()
            optimizer.step()
            print('Epoch:%3d'%(epoch),'loss:%.4f'%(loss))


        if np.abs(float(loss) - prev_loss) / float(loss) < args.convergence or accumulated_rounds >= args.ending_rounds:
            flag = False
        else:
            prev_loss = float(loss)

        for p in model.parameters():
            p.requires_grad = False
        model.r.requires_grad = True
        for epoch in range(1, 20):
            with torch.no_grad():
                z = model.encode(x, edge_index)
                pred_adj_list = []
                for node_j in np.arange(data.num_nodes):
                    one_line_edge_index = torch.cat(
                        [
                            torch.ones(data.num_nodes, dtype=torch.long, device=x.device)*node_j,
                            torch.arange(data.num_nodes, dtype=torch.long, device=x.device)
                        ]
                    ).view(2,-1) 
                    struct_output, _ = model.decoder(z, one_line_edge_index)
                    pred_adj_list.append(struct_output.view(-1))

            pred_adj = torch.cat(pred_adj_list).view(-1,data.num_nodes).detach().cpu()
            pred = pred_adj.numpy()

            # structure error
            mean_error_struct = np.abs((y-pred).mean(axis=1))

            # attribute error
            _, pred_attr = model.decoder(z, data.edge_index)
            mean_error_feature = torch.square(pred_attr - x).mean(dim=1).detach().cpu().numpy()

            # mixed error
            total_error = beta_1 * mean_error_struct + beta_2 * mean_error_feature

            # add supmodular function
            threshold = np.percentile(total_error, q=args.q)
            smgnn = SuperModularModel(threshold=threshold)#ModularModel(threshold=threshold) 
            anomaly_scores, residal_data, anomaly_nodes_index = smgnn(data, total_error)
            # roc auc score
            continuous_predicted_value = torch.zeros(data.num_nodes)
            for i in range(residal_data.batch.max()+1):
                continuous_predicted_value[anomaly_nodes_index[residal_data.batch == i]] = anomaly_scores.view(-1)[i]


            continuous_predicted_value = continuous_predicted_value.to(device)
            sp_loss = torch.maximum(
                        torch.zeros(continuous_predicted_value.size(0), device= residal_data.x.device), 
                        (model.r-continuous_predicted_value).view(-1)
                     ).sum() + model.r**2
            sp_loss.backward()
            optimizer_sp.step()
            #print('model.r',model.r)

        for p in model.parameters():
            p.requires_grad = True
        print('roc_auc_score:%.3f'%(roc_auc_score(anomaly_flag, continuous_predicted_value.cpu().numpy())))



