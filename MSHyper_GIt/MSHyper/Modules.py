
import math
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.data import data as D
from typing import Optional
from torch import Tensor



class ScaledDotProductAttention1(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.hatt1 = HypergraphConv(32, 1)
        self.hatt2 = HypergraphConv(1, 32)
        self.liear = torch.nn.Linear(32, 1)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # if mask is not None:
        #     attn = attn.masked_fill(mask, -1e9)
        # x = F.leaky_relu(self.hatt1(attn, mask), 0.2)
        # x = F.leaky_relu(self.hatt2(x, mask), 0.2)
        # attn = F.leaky_relu(self.liear(x))
        # output = torch.matmul(attn, v)


        x = self.hatt1(attn, mask)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.hatt2(x, mask)
        attn = F.leaky_relu(self.liear(x))
        # print(attn.shape)
        # output = torch.matmul(attn, v)
        return v, attn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):##temperature=11.3137084对于128维度进行了根号d
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = GCNConv(dataset_cora.num_node_features, 168)
        # self.conv2 = GCNConv(168, dataset_cora.num_classes)
        # self.conv1 = HypergraphConv(1, 16)
        # self.conv2 = HypergraphConv(16, 2)
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.softmax(x, dim=1)

        return x

class HypergraphConv(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{D}^{-1} \mathbf{H} \mathbf{W}
        \mathbf{B}^{-1} \mathbf{H}^{\top} \mathbf{X} \mathbf{\Theta}

    where :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` is the incidence
    matrix, :math:`\mathbf{W}` is the diagonal hyperedge weight matrix, and
    :math:`\mathbf{D}` and :math:`\mathbf{B}` are the corresponding degree
    matrices.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_attention=False,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(HypergraphConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def __forward__(self,
                    x,
                    hyperedge_index,
                    hyperedge_weight=None,
                    alpha=None):

        if hyperedge_weight is None:
            D = degree(hyperedge_index[0], x.size(0), x.dtype)
        else:
            D_1 = scatter_add(
                hyperedge_weight[hyperedge_index[1, 0:13263]],
                hyperedge_index[0, 0:13263],
                dim=0,
                dim_size=x.size(0))
            D_2 = scatter_add(
                hyperedge_weight[hyperedge_index[1, 13264:112859]],
                hyperedge_index[0, 13264:112859],
                dim=0,
                dim_size=x.size(0))
            D = torch.cat((D_1, D_2), dim=0)
            # ---------------------------------------------------------
        D = 1.0 / D
        D[D == float("inf")] = 0


        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B_1 = 1.0 / degree(hyperedge_index[1, 0:13263], 2708, x.dtype)
        B_2 = 1.0 / degree(hyperedge_index[1, 13264:], int(num_edges/2), x.dtype)
        B = torch.cat((B_1, B_2), dim=0)
        # ---------------------------------------------------------

        B[B == float("inf")] = 0
        if hyperedge_weight is not None:
            # B = B * hyperedge_weight; two next line is added by myself
            B = B * hyperedge_weight.t()
            B = B.t()

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        return out

    def message(self, x_j, edge_index_i, norm, alpha):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j.view(
            -1, self.heads, self.out_channels)
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out
        return out

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        r"""
        Args:
            x (Tensor): Node feature matrix :math:`\mathbf{X}`
            hyper_edge_index (LongTensor): Hyperedge indices from
                :math:`\mathbf{H}`.
            hyperedge_weight (Tensor, optional): Sparse hyperedge weights from
                :math:`\mathbf{W}`. (default: :obj:`None`)
        """
        x = torch.matmul(x, self.weight)
        # hyperedge_weight=data.edge_attr
        alpha = None

        if self.use_attention:
            x = x.view(-1, self.heads, self.out_channels)
            x_i, x_j = x[hyperedge_index[0]], x[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.__forward__(x, hyperedge_index, hyperedge_weight, alpha)

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

