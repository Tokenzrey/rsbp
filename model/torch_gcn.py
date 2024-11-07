import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 normalization='none'):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        # Input layer
        self.layers.append(GCNConv(in_channels=in_feats, out_channels=n_hidden, normalize=(normalization != 'none')))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(GCNConv(in_channels=n_hidden, out_channels=n_hidden, normalize=(normalization != 'none')))
        
        # Output layer
        self.layers.append(GCNConv(in_channels=n_hidden, out_channels=n_classes, normalize=(normalization != 'none')))

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, edge_index, edge_weight=edge_weight)
            if i != len(self.layers) - 1:  # Apply activation only on hidden layers
                h = self.activation(h)
        return h
