r"""modified from torch_geometric"""
import torch
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU
from torch_geometric.nn import MessagePassing

class MLPDecoder(MessagePassing):
    def __init__(self, in_channels, hidden_channels, output_channels, num_layers=2):
        super(MLPDecoder, self).__init__(aggr=None)
        # for reconstruct edges
        self.lin_start = Linear(in_channels*2, hidden_channels)
        self.lin_end = Linear(hidden_channels, 1)
        self.lin_end_1 = Linear(hidden_channels, hidden_channels)
        self.lin_end_2 = Linear(hidden_channels, 1)
        self.lin_list = ModuleList()
        for _ in range(num_layers):
            self.lin_list.append(Linear(hidden_channels, hidden_channels))
            
        # for reconstruct node attributes
        self.nodes_lin_start = Linear(in_channels, hidden_channels)
        self.nodes_lin_end = Linear(hidden_channels, output_channels)
        self.nodes_lin_list = ModuleList()
        for _ in range(num_layers):
            self.nodes_lin_list.append(Linear(hidden_channels, hidden_channels))

    def forward(self, x, edge_index, sigmoid=True):
        out = self.message(x[edge_index[0]], x[edge_index[1]])
        attr = self.nodes_lin_start(x)
        for block in self.nodes_lin_list:
            attr = block(attr)
        attr = self.nodes_lin_end(attr)
        return torch.sigmoid(out) if sigmoid else out, attr

    def message(self, x_i, x_j):
        temp = torch.cat([x_i,x_j], dim=1)
        temp = self.lin_start(temp).relu()
        for lin in self.lin_list:
            temp = lin(temp).relu()
        
        temp = self.lin_end_1(temp).relu()
        temp = self.lin_end_2(temp)
        return temp
    
class InnerProductDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, output_channels, num_layers=2):
        super(InnerProductDecoder, self).__init__()
            
        # for reconstruct node attributes
        self.nodes_lin_start = Linear(in_channels, hidden_channels)
        self.nodes_lin_end = Linear(hidden_channels, output_channels)
        self.nodes_lin_list = ModuleList()
        for _ in range(num_layers):
            self.nodes_lin_list.append(Linear(hidden_channels, hidden_channels))
        
        
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        
        attr = self.nodes_lin_start(z)
        for block in self.nodes_lin_list:
            attr = block(attr)
        attr = self.nodes_lin_end(attr)
        return torch.sigmoid(value) if sigmoid else value, attr