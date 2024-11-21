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
        # Debugging print statements
        print(f"Input feature tensor (x) type: {type(x)}, shape: {x.shape if isinstance(x, torch.Tensor) else 'N/A'}, device: {x.device}")
        print(f"Edge index content: {edge_index}")
        print(f"Edge index type: {type(edge_index)}, shape: {edge_index.edge_index.shape if hasattr(edge_index, 'edge_index') else 'N/A'}")
        if edge_weight is not None:
            print(f"Edge weight type: {type(edge_weight)}, shape: {edge_weight.shape if isinstance(edge_weight, torch.Tensor) else 'N/A'}, device: {edge_weight.device}")
        else:
            print("No edge weight provided.")
        
        # Ensure edge_index is in the correct format and on the same device as inputs
        edge_index = edge_index.edge_index
        
        h = x
        print(f"Initial node features (h) type: {type(h)}, shape: {h.shape if isinstance(h, torch.Tensor) else 'N/A'}, device: {h.device}")
        
        for i, layer in enumerate(self.layers):
            print(f"Processing layer {i + 1} of {len(self.layers)}")
            
            if i != 0:
                h = self.dropout(h)
            
            # Apply the GCNConv layer
            h = layer(h, edge_index, edge_weight=edge_weight)
            print(f"Output of layer {i + 1} type: {type(h)}, shape: {h.shape if isinstance(h, torch.Tensor) else 'N/A'}, device: {h.device}")
            
            if i != len(self.layers) - 1:  # Apply activation only on hidden layers
                h = self.activation(h)
                print(f"Activation applied on layer {i + 1}, shape: {h.shape}")
        
        return h
