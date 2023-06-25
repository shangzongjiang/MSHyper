from torch_geometric.datasets import Planetoid
# import torch
# dataset_cora = Planetoid(root='./cora/',name='Cora')
# print(dataset_cora)
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
import numpy as np
import math


# dataset_cora = Planetoid(root='./cora/',name='Cora')
# print(dataset_cora)
# print(len(dataset_cora))
# print(dataset_cora.num_classes)


class HypergraphConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_attention=True,
                 heads=6,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=False):
        super(HypergraphConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.fc = nn.Linear(self.out_channels * heads, self.out_channels)

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
        # print(hyperedge_index[0])
        bb = hyperedge_index  ####[2,498]
        # print(bb)
        aa = hyperedge_index[0]  ###[223][0是节点编号 1是边的编号]
        # print(aa)
        dd = x.size(0)  ###[223]
        cc = x.size()  ###[223,6,16]
        # print(x.size(0))
        # print(x.size())
        # print(x)
        if hyperedge_weight is None:
            #####下面D的含义是根据超边索引和节点个数计算每个节点的度
            D = degree(hyperedge_index[0], x.size(0), x.dtype)  ####D=[224]
            bbb = hyperedge_index[1, 0:4]
            # print(bbb)
            # ccc=hyperedge_weight[hyperedge_index[1, 0:13263]]
            # print(ccc)

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
        ccc = hyperedge_index[1].max().item() + 1  #####输出边的数量

        num_edges = 2 * (hyperedge_index[1].max().item() + 1)  ####输出两倍边的数量
        #通过degree计算超边的度，degree输入值为每个边连接节点的索引值以及边的总数
        B_test=degree(hyperedge_index[1],  ccc, x.dtype)
        B_1 = 1.0 / degree(hyperedge_index[1],  ccc, x.dtype)  ####B_1[2708]
        # B_2 = 1.0 / degree(hyperedge_index[1, 13264:], int(num_edges / 2), x.dtype)  ####B_2[4]
        # B = torch.cat((B_1, B_2), dim=0)  ####B=[2708]+[4]=[2714]
        # print(B)
        B=B_1
        # ---------------------------------------------------------

        B[B == float("inf")] = 0
        if hyperedge_weight is not None:
            # B = B * hyperedge_weight; two next line is added by myself
            B = B * hyperedge_weight.t()
            B = B.t()

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=D, alpha=alpha)
        # print(out)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        # print(out)
        return out

    def message(self, x_j, edge_index_i, norm, alpha):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j.view(
            -1, self.heads, self.out_channels)
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out
        return out

    def forward(self, x, hyperedge_index, hyperedge_weight=None):

        bbb = self.weight  #####权重系数[512,16x6]
        aaa = x  ####[223,512]
        x = torch.matmul(x, self.weight)  ##矩阵乘法x 和 权重系数相乘[223,512]x[512,96]--[223,96]
        # hyperedge_weight=data.edge_attr
        alpha = None

        if self.use_attention:
            a = x
            x = x.view(-1, self.heads, self.out_channels)  ###[223,6,16]第一维不变，第二维和第三维head x out_channels
            b = hyperedge_index[0] #####b=498
            # print(b)
            c = x[hyperedge_index[0]]
            # print(c[223,:,:])

            ######hyperedge_index第一行是节点索引值，第二行是超边索引值
            x_i, x_j = x[hyperedge_index[0]], x[hyperedge_index[1]]  ####x_i,x_j=[224,2,16]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  ###alpha=[224,2]
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)  ###alpha=[224,2]

        out = self.__forward__(x, hyperedge_index, hyperedge_weight, alpha)  ###out=[224,2,16]

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)  ####out=[224,32]
            out = self.fc(out)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = GCNConv(dataset_cora.num_node_features, 168)
        # self.conv2 = GCNConv(168, dataset_cora.num_classes)
        self.conv1 = HypergraphConv(512, 16)
        self.conv2 = HypergraphConv(16, 223)
        # self.conv1 = GCNConv(1, 16)
        # self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        ####x=[223,512] edge_index=[2,223]
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)  #####x=[224,16]
        x = F.dropout(x, training=self.training)  ###x=[224,16]
        x = self.conv2(x, edge_index)
        x = F.softmax(x, dim=1)

        return x


def get_mask(input_size, window_size, inner_size, device):  ###inner_size:3 input_size 169 window_size[4,4,4]
    """Get the attention mask of PAM-Naive"""
    # Get the size of all layers
    all_size = []
    all_size.append(input_size)  ###all_size=169
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])  ##math.floor向下取整 layer_size=[169,42,10,2]
        all_size.append(layer_size)
        # print(all_size)

    seq_length = sum(all_size)  ####seq_length=169+42+10+2=223

    # get inter-scale mask

    j = 4
    aaa = []
    num_all = []
    index_all = []
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):  ###第一层[169,211]
            left_side = (start - all_size[layer_idx - 1]) + (i - start) * window_size[layer_idx - 1]
            if i == (start + all_size[layer_idx] - 1):
                right_side = start
            else:
                right_side = (start - all_size[layer_idx - 1]) + (i - start + 1) * window_size[layer_idx - 1]
            #####得到尺度间数据列表
            num = list(range(left_side, right_side))
            num += list([i])
            num_all += num
            #####得到尺度间数据边的索引
            index = list(np.repeat(j, len(num)))
            index_all += index
            # print(mask[[5],:])#####输出第五行
            j += 1
            # mask[i, left_side:right_side] = j
            # mask[left_side:right_side, i] = j####第0行到第4行为同一个父节点i
    conect_result = np.vstack((num_all, index_all))
    # mask
    # print(mask)

    # mask = (1 - mask).bool()
    # mask

    return conect_result



model = Net()
print(model)
seq_length = 223
small = [int(i) for i in range(0, seq_length)]  ###small=223
# print(len(small))
mask_edge = small
small1 = [int(i) for i in range(0, seq_length)]  ####small1=223
for i in range(seq_length):
    if i <= 168:
        mask_edge[i] = 0
    elif i > 168 and i <= 209:
        mask_edge[i] = 1
    elif i > 209 and i <= 219:
        mask_edge[i] = 2
    else:
        mask_edge[i] = 3
edge_index = np.vstack((small1, mask_edge))  ###对数据按行拼接


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

# print(data)
# print(data.shape)

input_size = 169
window_size = [4, 4, 4]
inner_size = 3

####把数据转为tensor类型
edge_index = torch.tensor(edge_index, dtype=torch.long)####edge_index=[2,223]
intra_mask = get_mask(input_size, window_size, inner_size, device)####intra_mask=[2,275]
edge_new = np.hstack((edge_index, intra_mask))  #####将intra_mask接到edge_index后面  edge_new=[2,498]
edge_new = torch.tensor(edge_new, dtype=torch.long)

x = np.random.randint(0, 100, (223, 512))  ####从0-100里选数生成(223,512)shape的数据
x = torch.tensor(x, dtype=torch.float)
y = mask_edge   #y=223
y = torch.tensor(y)
train_mask = [bool(i) for i in range(1, seq_length + 1)]
edge_attr = torch.tensor([10, 20, 30], dtype=torch.float)
# print(train_mask)
# train_mask=bool(train_mask)
# print(y)
# train_mask=torch.tensor(train_mask.bool(),dtype=torch.bool)
train_mask = torch.tensor(train_mask, dtype=torch.bool)
val_mask = train_mask
test_mask = train_mask
data = D.Data()

data.x, data.y, data.edge_index, data.edge_attr, data.train_mask, data.val_mask, data.test_mask \
    = x, y, edge_new, edge_attr, train_mask, val_mask, test_mask

data = data.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, pred = torch.max(out[data.train_mask], dim=1)
    correct = (pred == data.y[data.train_mask]).sum().item()
    acc = correct / data.train_mask.sum().item()

    print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(
        epoch, loss.item(), acc))
model.eval()
out = model(data)
loss = criterion(out[data.test_mask], data.y[data.test_mask])
_, pred = torch.max(out[data.test_mask], dim=1)
correct = (pred == data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum().item()
print("test_loss: {:.4f} test_acc: {:.4f}".format(loss.item(), acc))


class GCNConv(MessagePassing):
    def __init__(self, input_dim, output_dim):
        super(GCNConv, self).__init__(aggr='add')
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(
            edge_index=edge_index, num_nodes=x.shape[0])

        x = self.fc(x)

        row, col = edge_index
        deg = degree(index=col, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        norm = norm.view(-1, 1)
        m = norm * x_j
        return m

    def update(self, aggr_out):
        return aggr_out

