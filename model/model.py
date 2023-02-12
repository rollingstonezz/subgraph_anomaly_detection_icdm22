import torch
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from typing import List, Optional, Set, Callable, get_type_hints
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from torch_geometric.typing import Adj, Size
from torch import Tensor

class GAE(torch.nn.Module):
    def __init__(self, encoder, decoder=None, r=10.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.r = Parameter(torch.Tensor(1))
        GAE.reset_parameters(self, r)

    def reset_parameters(self, r):
        reset(self.encoder)
        reset(self.decoder)
        torch.nn.init.constant_(self.r, r)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None, x=None, loss_type='l2', beta=1.0):
        
        loss_function = F.l1_loss if loss_type == 'l1' else F.mse_loss
        
        # positive
        pred_pos_struct, pred_pos_attr = self.decoder(z, pos_edge_index)
        pred_pos_struct = pred_pos_struct.view(-1)
        pos_loss_struct = loss_function(
            pred_pos_struct,
            torch.ones(len(pos_edge_index[0]), device=z.device)
        )
        #loss_attr = loss_function(pred_pos_attr, x)
        loss_attr = ((pred_pos_attr - x) ** 2).mean(dim=1)
        
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0), num_neg_samples=len(pos_edge_index[0]))
            neg_edge_index = neg_edge_index.to(z.device)
            
        pred_neg_struct, pred_neg_attr = self.decoder(z, neg_edge_index)
        pred_neg_struct = pred_neg_struct.view(-1)

        neg_loss_struct = loss_function(
            pred_neg_struct, 
            torch.zeros(len(neg_edge_index[0]), device=z.device)
        )
        ratio = len(neg_edge_index[0]) / len(pos_edge_index[0]) 
        
        return pos_loss_struct, neg_loss_struct*ratio, loss_attr

    def test(self, z, pos_edge_index, neg_edge_index):

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred, _ = self.decoder(z, pos_edge_index)
        neg_pred, _ = self.decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)
