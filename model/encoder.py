r"""modified from torch_geometric"""
import torch
from torch_geometric.nn import GCNConv, EdgeConv
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU

class MLPLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(MLPLayer, self).__init__()
        self.lin_start = Linear(in_channels, hidden_channels)
        self.lin_end = Linear(hidden_channels, out_channels)
        self.lin_list = ModuleList()
        for _ in range(num_layers):
            self.lin_list.append(Linear(hidden_channels, hidden_channels))
            
    def forward(self, x):
        x = self.lin_start(x).relu()
        for lin in self.lin_list:
            x = lin(x).relu()
        x = self.lin_end(x)
        return x

class EdgeConvEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, aggr='add'):
        super(EdgeConvEncoder, self).__init__()
        self.num_layers = num_layers
        if num_layers == -1:
            self.conv1 = EdgeConv(MLPLayer(in_channels*2, hidden_channels, out_channels), aggr=aggr)
        else:
            self.conv1 = EdgeConv(MLPLayer(in_channels*2, hidden_channels, hidden_channels), aggr=aggr)
            self.conv2 = EdgeConv(MLPLayer(hidden_channels*2, hidden_channels, out_channels), aggr=aggr)
            self.conv_list = ModuleList()
            for _ in range(num_layers-2):
                self.conv_list.append(EdgeConv(MLPLayer(hidden_channels*2, hidden_channels, hidden_channels), aggr=aggr))

    def forward(self, x, edge_index, batch=None):
        if self.num_layers == -1:
            x = self.conv1(x, edge_index)
            return x
        else:
            x = self.conv1(x, edge_index).relu()
            for conv in self.conv_list:
                x = conv(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x
        
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, aggr='add'):
        super(GCNEncoder, self).__init__()
        self.num_layers = num_layers
        if num_layers == -1:
            self.conv1 = GCNConv(in_channels, out_channels)
        else:
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels) 
            self.conv_list = ModuleList()
            for _ in range(num_layers-2):
                self.conv_list.append(GCNConv(hidden_channels, hidden_channels))

    def forward(self, x, edge_index, batch=None):
        if self.num_layers == -1:
            x = self.conv1(x, edge_index)
            return x
        else:
            x = self.conv1(x, edge_index).relu()
            for conv in self.conv_list:
                x = conv(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x