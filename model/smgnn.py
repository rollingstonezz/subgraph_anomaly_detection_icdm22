import torch
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.utils import subgraph 
from torch.autograd import Variable
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter
from networkx.algorithms.components import strongly_connected_components

from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch import Tensor
import networkx as nx

class ModularGNN(MessagePassing):
    def __init__(self, nn, aggr='add', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.readout = aggr

    def forward(self, x, edge_index, batch=None):
        """"""
        if batch is None:
            batch = torch.zeros(len(x), dtype=torch.long, device=x.device) 
            
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        
        out = x[1]
        out = scatter(out, batch, dim=0, reduce=self.readout) 

        return out

    def message(self, x_i, x_j):
        return x_j
    
class ModularModel(torch.nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.smgnn = ModularGNN(nn=torch.sum)
        
    def forward(self, data, mean_error_list):
        
        #generate subgraph by threshold
        residual_edge_index, _ = subgraph(
            torch.arange(data.num_nodes)[mean_error_list>self.threshold], 
            data.edge_index, 
            relabel_nodes=True
        )
        residal_data = Data(
            x=torch.tensor(mean_error_list[mean_error_list>self.threshold]).view(-1,1),
            edge_index=residual_edge_index,
        )
        
        # find components
        residual_nx = to_networkx(
            residal_data, 
            to_undirected=True, 
            remove_self_loops=True
        )
        batch = torch.zeros(residal_data.num_nodes, dtype=torch.long)
        i = 0
        for comp in nx.connected_components(residual_nx):
            batch[list(comp)] = i
            i += 1
        residal_data.batch = batch
        residal_data.to(data.x.device)
        
        # compute anoamly scores
        anomaly_scores = self.smgnn(residal_data.x, residal_data.edge_index, residal_data.batch)
        
        return anomaly_scores, residal_data, torch.arange(len(mean_error_list))[mean_error_list>self.threshold]

class SuperModularModel(torch.nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.smgnn = SuperModularGNN(nn=torch.sum)
        
    def forward(self, data, mean_error_list):
        
        #generate subgraph by threshold
        residual_edge_index, _ = subgraph(
            torch.arange(data.num_nodes)[mean_error_list>self.threshold], 
            data.edge_index, 
            relabel_nodes=True
        )
        residal_data = Data(
            x=torch.tensor(mean_error_list[mean_error_list>self.threshold]).view(-1,1),
            edge_index=residual_edge_index,
        )
        
        # find components
        residual_nx = to_networkx(
            residal_data, 
            to_undirected=True, 
            remove_self_loops=True
        )
        batch = torch.zeros(residal_data.num_nodes, dtype=torch.long)
        i = 0
        for comp in nx.connected_components(residual_nx):
            batch[list(comp)] = i
            i += 1
        residal_data.batch = batch
        residal_data.to(data.x.device)
        
        # compute anoamly scores
        anomaly_scores = self.smgnn(residal_data.x, residal_data.edge_index, residal_data.batch)
        
        return anomaly_scores, residal_data, torch.arange(len(mean_error_list))[mean_error_list>self.threshold]
    
class SuperModularGNN(MessagePassing):
    def __init__(self, nn, aggr='add', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.readout = aggr

    def forward(self, x, edge_index, batch=None):
        """"""
        if batch is None:
            batch = torch.zeros(len(x), dtype=torch.long, device=x.device) 
            
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        
        out = x[1]+self.propagate(edge_index, x=x, size=None)
        out = scatter(out, batch, dim=0, reduce=self.readout) 

        return out

    def message(self, x_i, x_j):
        return x_j
    
class WSSuperModularModel(torch.nn.Module):
    def __init__(self, data, mean_error_list, threshold=0.5, anomaly_scale_nodes=None):
        super().__init__()
        self.threshold = threshold
        self.nn1 = Sequential(
            Linear(1, data.num_features, bias=False),
            ReLU(),
            Linear(data.num_features, 1, bias=False),
        )
        self.nn2 = Sequential(
            Linear(1, data.num_features, bias=False),
            ReLU(),
            Linear(data.num_features, 1, bias=False),
        )
        self.smgnn = WSSuperModularGNN(nn1=self.nn1,nn2=self.nn2)
        
        self.inside_flag = torch.tensor([False]*data.num_nodes)
        self.inside_flag[anomaly_scale_nodes] = True
        
        self.threshold_flag = torch.tensor(mean_error_list>self.threshold)
        
        #generate two subgraph by threshold & inside scale
        # inside
        residual_edge_index_inside, _ = subgraph(
            torch.arange(data.num_nodes)[self.threshold_flag & self.inside_flag], 
            data.edge_index, 
            relabel_nodes=True
        )
        residal_data_inside = Data(
            x=torch.tensor(mean_error_list[self.threshold_flag & self.inside_flag]).view(-1,1),
            #x=torch.tensor(data.x[self.threshold_flag & self.inside_flag]),
            edge_index=residual_edge_index_inside,
        )
        
        # find components
        residual_nx_inside = to_networkx(
            residal_data_inside, 
            to_undirected=True, 
            remove_self_loops=True
        )
        batch = torch.zeros(residal_data_inside.num_nodes, dtype=torch.long)
        i = 0
        for comp in nx.connected_components(residual_nx_inside):
            batch[list(comp)] = i
            i += 1
        residal_data_inside.batch = batch
        residal_data_inside.to(data.x.device)
        self.residal_data_inside = residal_data_inside
        
        
        # outside
        residual_edge_index_outside, _ = subgraph(
            torch.arange(data.num_nodes)[self.threshold_flag & (~self.inside_flag)], 
            data.edge_index, 
            relabel_nodes=True
        )
        residal_data_outside = Data(
            x=torch.tensor(mean_error_list[self.threshold_flag & (~self.inside_flag)]).view(-1,1),
            #x=torch.tensor(data.x[self.threshold_flag & (~self.inside_flag)]),
            edge_index=residual_edge_index_outside,
        )
        
        # find components
        residual_nx_outside = to_networkx(
            residal_data_outside, 
            to_undirected=True, 
            remove_self_loops=True
        )
        batch = torch.zeros(residal_data_outside.num_nodes, dtype=torch.long)
        i = 0
        for comp in nx.connected_components(residual_nx_outside):
            batch[list(comp)] = i
            i += 1
        residal_data_outside.batch = batch
        residal_data_outside.to(data.x.device)
        self.residal_data_outside = residal_data_outside
        
    def forward(self):
        
        # compute anoamly scores
        anomaly_scores_inside = self.smgnn(self.residal_data_inside.x, self.residal_data_inside.edge_index, self.residal_data_inside.batch)
        anomaly_scores_outside = self.smgnn(self.residal_data_outside.x, self.residal_data_outside.edge_index, self.residal_data_outside.batch)
        
        return anomaly_scores_inside, anomaly_scores_outside
    
class WSSuperModularGNN(MessagePassing):
    def __init__(self, nn1, nn2, aggr='add', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn1 = nn1
        self.nn2 = nn2
        self.readout = aggr

    def forward(self, x, edge_index, batch=None):
        """"""
        if batch is None:
            batch = torch.zeros(len(x), dtype=torch.long, device=x.device) 
            
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        
        out = self.nn2(x[1])+self.propagate(edge_index, x=x, size=None)
        out = scatter(out, batch, dim=0, reduce=self.readout) 
        print(out.shape)

        return out

    def message(self, x_i, x_j):
        return self.nn1(x_j)
    
class weightConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            print("Entered")
            w=module.weight.data
            w=w.clamp(0.0)
            module.weight.data=w
    