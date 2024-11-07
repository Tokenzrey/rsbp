import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GraphConvEdgeWeight(MessagePassing):
    def __init__(self, in_channels, out_channels, allow_zero_in_degree=False, norm="both", bias=True, activation=None):
        super(GraphConvEdgeWeight, self).__init__(aggr='add')  # PyG uses 'add' by default for summing messages.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.allow_zero_in_degree = allow_zero_in_degree
        self.norm = norm
        self.activation = activation

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        if not self.allow_zero_in_degree and torch.any(degree(edge_index[1]) == 0):
            raise ValueError('There are 0-in-degree nodes in the graph. '
                             'This is harmful for some applications, '
                             'causing silent performance regression. '
                             'Consider adding self-loops or setting allow_zero_in_degree to True.')

        # Add self-loops to handle 0-in-degree nodes
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1.0, num_nodes=x.size(0))

        # Normalize node features
        if self.norm == 'both':
            row, col = edge_index
            deg = degree(row, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        elif self.norm == 'left':
            row, col = edge_index
            deg = degree(row, x.size(0), dtype=x.dtype)
            deg_inv = deg.pow(-1)
            norm = deg_inv[row] * edge_weight
        else:
            norm = edge_weight

        # Apply linear transformation
        x = torch.matmul(x, self.weight)

        # Start propagating messages
        out = self.propagate(edge_index, x=x, edge_weight=norm)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias

        # Apply activation if present
        if self.activation is not None:
            out = self.activation(out)

        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
