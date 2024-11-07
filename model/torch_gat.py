import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop=0,
                 attn_drop=0,
                 negative_slope=0.2,
                 residual=False):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        # Input layer
        self.gat_layers.append(GATConv(
            in_channels=in_dim,
            out_channels=num_hidden,
            heads=heads[0],
            dropout=attn_drop,
            negative_slope=negative_slope,
            concat=True,  # For multi-head, concatenate output from each head
            bias=True,
        ))

        # Hidden layers
        for l in range(1, num_layers):
            # The input dimension for hidden layers = num_hidden * num_heads (from previous layer)
            self.gat_layers.append(GATConv(
                in_channels=num_hidden * heads[l - 1],
                out_channels=num_hidden,
                heads=heads[l],
                dropout=attn_drop,
                negative_slope=negative_slope,
                concat=True,
                bias=True,
            ))

        # Output layer
        self.gat_layers.append(GATConv(
            in_channels=num_hidden * heads[-2],
            out_channels=num_classes,
            heads=heads[-1],
            dropout=attn_drop,
            negative_slope=negative_slope,
            concat=False,  # Do not concatenate at the final layer
            bias=True,
        ))

    def forward(self, x, edge_index):
        h = x
        # Apply GAT layers
        for l in range(self.num_layers):
            h = self.gat_layers[l](h, edge_index)
            if l != self.num_layers - 1:
                h = h.flatten(1)  # Flatten for multi-head concat and apply activation
                if self.activation is not None:
                    h = self.activation(h)
        
        # Output layer (no activation)
        logits = self.gat_layers[-1](h, edge_index).mean(1)
        return logits
