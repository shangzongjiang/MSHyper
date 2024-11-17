import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.data import data as D
from torch.nn import Linear
import torch_scatter
from math import sqrt
from .Layers import EncoderLayer, Decoder, Predictor
from .Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from .Layers import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
from .embed import DataEmbedding, CustomEmbedding,DataEmbedding_new

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        configs.device = torch.device("cuda")
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        """
        #####之前代码
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Tran = nn.Linear(self.pred_len, self.pred_len)
        """

            # self.Linear_Tran.weight = nn.Parameter(
            #     (1 / (self.pred_len)) * torch.ones([self.pred_len, self.pred_len]))
        #以下为超图设计代码
        self.individual = configs.individual
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Tran = nn.Linear(self.pred_len, self.pred_len)
        self.enc_embedding=DataEmbedding(configs.enc_in, configs.d_model, configs.dropout)
        self.mask, self.all_size, self.adj= get_mask(configs.seq_len, configs.window_size, configs.inner_size, configs.khop, configs.device)
        self.Ms_length = sum(self.all_size)
        self.conv_layers = eval(configs.CSCM)(configs.enc_in, configs.window_size, configs.d_bottleneck)
        self.conv1 = HypergraphConv(configs)
        # self.conv2 = TMP(128, 512)
        self.convtra = nn.Linear(self.channels, self.channels)
        self.convtra.weight = nn.Parameter(
            (1 / (self.channels)) * torch.ones([self.channels, self.channels]))
        self.out_tran = nn.Linear(self.Ms_length, self.pred_len)

        self.predictor=Predictor(self.channels,self.pred_len*self.channels)
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.indexes=refer_points(self.all_size, configs.window_size, configs.device)

    def forward(self, x,x_mark_enc):
        #instance_normal
        mean_enc=x.mean(1,keepdim=True).detach()
        std_enc=torch.sqrt(torch.var(x,dim=1,keepdim=True,unbiased=False)+1e-5).detach()
        x = x - mean_enc
        x=x / std_enc
        _, _, N = x.shape
        #以下为超图代码
        # seq_enc = self.enc_embedding(x, x_mark_enc)
        mask=self.mask
        # mask = self.mask.repeat(len(x), 1, 1)
        mask = torch.tensor(mask).to(x.device)
        seq_enc = self.conv_layers(x).to(x.device)
        seq_enc = torch.tensor(seq_enc, dtype=torch.float).to(x.device)

        # x_out = self.conv1(seq_enc, mask)
        # x_out=self.chan_tran(x_out)
        adj = self.adj
        hw = None
        x_out = self.conv1(seq_enc, mask, adj, hw)
        # x = self.conv2(seq_enc, mask, adj, hw)
        # ####自己实现金字塔最后一层
        # indexes = self.indexes.repeat(x.size(0), 1, 1, x.size(2)).to(x.device)
        # indexes = indexes.view(x_out.size(0), -1, x_out.size(2))
        # all_enc = torch.gather(x_out, 1, indexes)
        # seq_enc = all_enc.view(x.size(0), self.all_size[0], -1).permute(0, 2, 1)
        # out=x[:,-1,:]
        # out_tra=self.predictor(out).view(x.size(0),self.pred_len,-1).permute(0,2,1)
        ###融合模块
        if self.individual:
            ####自己实现金字塔最后一层
            indexes = self.indexes.repeat(x.size(0), 1, 1, x.size(2)).to(x.device)
            indexes = indexes.view(x_out.size(0), -1, x_out.size(2))
            all_enc = torch.gather(x_out, 1, indexes)
            seq_enc = all_enc.view(x.size(0), self.all_size[0], -1).permute(0, 2, 1)
            out = x[:, -1, :]
            out_tra = self.predictor(out).view(x.size(0), self.pred_len, -1).permute(0, 2, 1)
            x = self.Linear(x.permute(0,2,1))
        else:
            # x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            # x_out=self.out_tran(x_out.permute(0,2,1)).permute(0,2,1)
            x = self.Linear(x.permute(0, 2, 1))
            out_tra = self.out_tran(x_out.permute(0, 2, 1))
            # x_out=self.out_tran(x_out.permute(0,2,1))
        # x = x + x_out


        x = x + out_tra

        # x = self.Linear_Tran(seq_enc)

        # x=x.permute(0,2,1)

        x=self.Linear_Tran(x).permute(0,2,1)
        # x = x.permute(0, 2, 1)
        # x = self.convtra(x)
        # x = self.Linear_Tran(x)
        # out=self.projector(x).permute(0, 2, 1)[:, :, :N]

        # result = out_tra * std_enc + mean_enc
        x = x * std_enc+ mean_enc
        return x # [Batch, Output length, Channel]






class TMP(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_attention=True,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=False):
        super(TMP, self).__init__(aggr='add')
        self.soft=nn.Softmax(dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        # self.fc = nn.Linear(self.out_channels * heads, self.out_channels)
        self.W=nn.Linear(1,512)
        self.dropout1 = nn.Dropout(0.1)
        # self.dropout_rate=0.1
        ##超边到超边之间的attention计算




        # 超边到节点的信息聚合attention
        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            # self.weight = Parameter(
            #     torch.Tensor(in_channels, heads * out_channels))
            self.weight = Parameter(
                torch.Tensor(in_channels, out_channels))
            # self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * int(out_channels / heads)))
            # aaa=self.att
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            # self.bias = Parameter(torch.Tensor(heads * out_channels))
            self.register_parameter('bias', None)

        self.reset_parameters()


    #初始化权重和偏置参数
    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)
    ####计算超图的度，然后根据度计算超图的归一化权重
    def __forward__(self,
                    x,
                    hyperedge_index,adj,
                    hyperedge_weight=None,
                    alpha=None):
        # print('okk')
        # print(x.size(0))
        # bbb=x[1,:,:]
        # print(bbb)
        # print(x[2,:,:])
        aa = hyperedge_index[0]
        dd=x.size(0)
        # adj=adj
        # print(hyperedge_index[0][0].size(0))
        # bbb=hyperedge_index##[32,2,223]
        # aaa=hyperedge_index[-1,:,:]##[2,223]
        # cc=x    ###[32,223,512]
        # print(hyperedge_index[0][1])
        # print(x.szie(0))
        # print("okkk")

        # hyperedge_index=hyperedge_index[-1,:,:]
        # x=x[-1,:,:]

        if hyperedge_weight is None:
            #####D是节点的度
            D = degree(hyperedge_index[0], x.size(0), x.dtype)
            # hyperedge_weight = x.new_ones(0)
            # D = torch_scatter.scatter_add(hyperedge_weight[hyperedge_index[0]],
            #                 hyperedge_index[0], dim=0, dim_size=x.size(0))
            # D = degree(hyperedge_index[0], x)
        else:
            D_1 = torch_scatter.scatter_add(
                hyperedge_weight[hyperedge_index[1, 0:13263]],
                hyperedge_index[0, 0:13263],
                dim=0,
                dim_size=x.size(0))
            D_2 = torch_scatter.scatter_add(
                hyperedge_weight[hyperedge_index[1, 13264:112859]],
                hyperedge_index[0, 13264:112859],
                dim=0,
                dim_size=x.size(0))
            D = torch.cat((D_1, D_2), dim=0)
            # ---------------------------------------------------------
        D = 1.0 / D
        D[D == float("inf")] = 0


        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B_1 = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        # B_1 = 1.0 / degree(hyperedge_index[1], 2708, x.dtype)
        B_2 = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        # B_1 = 1.0 / degree(hyperedge_index[1, 0:13263], 2708, x.dtype)
        # B_2 = 1.0 / degree(hyperedge_index[1, 13264:], int(num_edges/2), x.dtype)
        # B = torch.cat((B_1, B_2), dim=0)
        B=B_1
        # ---------------------------------------------------------

        B[B == float("inf")] = 0
        if hyperedge_weight is not None:
            # B = B * hyperedge_weight; two next line is added by myself
            B = B * hyperedge_weight.t()
            B = B.t()

        #调用propagate方法执行消息传递，传递信息为节点特征和归一化权重
        ####propogate执行消息聚合和更新节点特征的操作
        ####propogate执行两遍，因为需要执行源节点到目标节点和目标节点到源节点
        ####输出结果为超图卷积结果
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        # print("okkkk")
        # alpha=torch.matmul(out1, out)
        # alpha = softmax(alpha)
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # outla=torch.matmul(alpha, x)
        return out


    ####message在消息传递中计算每个节点收到的消息
    #####将输入的节点特征和超边归一化权重相乘
    ####并根据头数和输出通道数将结果重新组织
    def message(self, x_j, edge_index_i, norm, alpha):
        # out = norm[edge_index_i].view(-1, 1, 1) * x_j.view(-1, self.heads, self.out_channels)###origional
        out = norm[edge_index_i].view(-1, 1, 1) * x_j####
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out
        return out
    ####forward是对__forward__方法的封装，传入了输入的节点特征和超图，返回超图卷积结果
    def forward(self, x, hyperedge_index, adj, hyperedge_weight=None):
        r"""
        Args:
            x (Tensor): Node feature matrix :math:`\mathbf{X}`
            hyper_edge_index (LongTensor): Hyperedge indices from
                :math:`\mathbf{H}`.
            hyperedge_weight (Tensor, optional): Sparse hyperedge weights from
                :math:`\mathbf{W}`. (default: :obj:`None`)
        """
        adj1=adj
        x = torch.matmul(x, self.weight)
        # hyperedge_weight=data.edge_attr
        alpha = None
        B,L,D=x.shape
        scale=1./sqrt(D)

        # if self.use_attention:
        #     x = x.view(-1,-1, self.heads, self.out_channels)####[32,223,512]--[7136,1,512]
        #     x_i, x_j = x[hyperedge_index[0]], x[hyperedge_index[1]]#####x_i,x_j[2,223,1,512]
        #     alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)####alpha=[2,223,1]
        #     alpha = F.leaky_relu(alpha, self.negative_slope)
        #     alpha = softmax(alpha, hyperedge_index[0], x.size(0))
        #     alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        adj = torch.tensor(adj).to(x.device)
        for i in range(x.size(0)):
            x_new = x[i, :, :]
            hyperedge_index_new = hyperedge_index[i, :, :]
            if self.use_attention:
                # aaa=x[i,:,:]#####[32,223,512]--[223,512]
                # x=x[i,:,:]
                # hyperedge_index=hyperedge_index[i,:,:]
                # print(x_new.size(0))
                x_new1 = x_new
                # x_new = x_new.view(-1, self.heads, self.out_channels)####[223,512]--[223,head,512]
                x_new = x_new.view(x_new.size(0), self.heads, -1) ####[223,512]--[223,head,512]
                # x_i, x_j = x_new[hyperedge_index_new[0]], x_new[hyperedge_index_new[1]]  #####x_i,x_j[2,223,1,512]
                x_i= x_new[hyperedge_index_new[0]]


                """
                ###add_start
                x_jnew=x_j.permute(0,2,1)
                alpha_new = torch.bmm(x_i, x_jnew)
                alpha_new=self.soft(alpha_new)
                alpha_new = F.dropout(alpha_new, p=self.dropout, training=self.training)
                x_last=torch.bmm(alpha_new,x_j).sum(dim=-1)
                ###add_end
                """

                edges=hyperedge_index_new[1]
                """
                # 重复聚合
                # 创建一个字典用于存储超边的聚合值
                aggregated_values = {}

                # 遍历节点、超边和节点值的张量，计算超边值的总和
                for edge in torch.unique(edges):
                    mask = edges == edge
                    edge_value = torch.sum(x_i[mask])
                    aggregated_values[edge.item()] = edge_value.item()

                # 根据超边连接的节点信息，计算超边的聚合值
                aggregated_values = [aggregated_values[edge.item()] for edge in edges]
                kkk=aggregated_values
                """
                #非重复聚合
                # node-edge
                # unique_edges = torch.unique(edges)
                unique_edges, inverse_indices = torch.unique(edges,return_inverse=True)
                aggregated_values = torch.stack([torch.sum(x_i[edges == edge]) for edge in unique_edges])
                # kkk=aggregated_values
                aggregated_values=aggregated_values.unsqueeze(0)
                # print(kkk.shape)
                ###超边-超边之间的attention计算
                # scores=torch.matmul(aggregated_values,self.W(aggregated_values).transpose(-2,-1))
                # attention_weights = (self.soft(torch.matmul(aggregated_values,adj)))
                attention_weights = (torch.softmax(aggregated_values * adj, dim=-1))
                # attention_weights=(torch.softmax(aggregated_values*adj,dim=-1))

                # attended_value = torch.einsum("bh,hh->bh",aggregated_values,attention_weights)
                attended_value = torch.einsum("hh,hb->hb", attention_weights, aggregated_values.T)
                attended_value=attended_value.to(torch.float32)
                attended_value = self.dropout1(self.W(attended_value))
                restored_edges = attended_value[inverse_indices]
                # xjj=restored_edges[hyperedge_index_new[1]]
                x_j=restored_edges.view(hyperedge_index_new.size(1), self.heads, -1)
                # aa=torch.cat([x_i, x_j], dim=-1)
                # x_j= x_j[hyperedge_index_new[0]]

                ###edge-node
                alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  ####alpha=[223,head]
                alpha = F.leaky_relu(alpha, self.negative_slope)
                alpha = softmax(alpha, hyperedge_index_new[0], x_new.size(0))
                # alpha = softmax(alpha, hyperedge_index_new[0], x_new.size(0))

                alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            # out_batch = self.__forward__(x[i,:,:], hyperedge_index[i,:,:], hyperedge_weight, alpha)
            out_batch = self.__forward__(x_new, hyperedge_index_new, adj, hyperedge_weight, alpha)###out_batch=[223,1024]
            # out_batch = out_batch.view(-1,1, self.heads * self.out_channels)  ####out=[224,32]
            out_batch = out_batch.view(-1, 1, self.out_channels)

            # out_batch = self.fc(out_batch)####原来需要加维度转换
            if i==0:
                out=out_batch
            else:
                out=torch.cat((out,out_batch),1)
        # out_test=out###[223,32,512]
        out=out.transpose(0,1)


        # out = self.__forward__(x, hyperedge_index, hyperedge_weight, alpha)

        # if self.concat is True:
        #     out = out.view(-1, self.heads * self.out_channels)
        # else:
        #     out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class Predictor(nn.Module):

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data):
        out = self.linear(data)
        out = out
        return out


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, self.W(key).transpose(-2, -1))  # 计算注意力分数

        if mask is not None:
            scores.masked_fill_(mask == 0, -float('inf'))  # 将无效位置的分数置为负无穷

        attention_weights = torch.softmax(scores, dim=-1)  # 计算注意力权重
        attended_value = torch.matmul(attention_weights, value)  # 根据权重对value进行加权求和

        return attended_value, attention_weights

class HypergraphConv(MessagePassing):
    def __init__(self, configs, use_attention=True, heads=1, concat=True, negative_slope=0.2, dropout=0.1, bias=False):
        super(HypergraphConv, self).__init__(aggr='add')
        self.soft=nn.Softmax(dim=0)
        self.in_channels = configs.enc_in
        self.d_model = configs.d_model
        self.out_channels = configs.dec_in
        self.use_attention = use_attention
        self.W = nn.Linear(1, configs.enc_in)
        self.dropout1 = nn.Dropout(0.1)
        # self.fc = nn.Linear(self.out_channels * heads, self.out_channels)
        ##hyperedge_attention
        self.W_query=nn.Linear(configs.enc_in,configs.enc_in)
        self.W_key=nn.Linear(configs.enc_in,configs.enc_in)
        self.W_value = nn.Linear(configs.enc_in,configs.enc_in)
        self.C=500


        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(torch.Tensor(self.d_model, configs.dec_in))
            # self.att1 = Parameter(torch.Tensor(1, heads, int(out_channels / heads)))
            self.att = Parameter(torch.Tensor(1, heads, 2 * int(configs.dec_in / heads)))###如果报错，要看输入输出维度是否能够对应
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(self.d_model, configs.dec_in))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * configs.dec_in))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(configs.dec_in))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    #初始化权重和偏置参数
    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)
    ####计算超图的度，然后根据度计算超图的归一化权重
    def __forward__(self, x, hyperedge_index, adj, hyperedge_weight=None, alpha=None):
        D = degree(hyperedge_index[0], x.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        # --------------------------------------------------------
        B[B == float("inf")] = 0

        #调用propagate方法执行消息传递，传递信息为节点特征和归一化权重
        ####propogate执行消息聚合和更新节点特征的操作
        ####propogate执行两遍，因为需要执行源节点到目标节点和目标节点到源节点
        ####输出结果为超图卷积结果
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        return out


    ####message在消息传递中计算每个节点收到的消息
    #####将输入的节点特征和超边归一化权重相乘
    ####并根据头数和输出通道数将结果重新组织
    def message(self, x_j, edge_index_i, norm, alpha):
        # out = norm[edge_index_i].view(-1, 1, 1) * x_j.view(-1, self.heads, self.out_channels)###origional
        out = norm[edge_index_i].view(-1, 1, 1) * x_j####
        if alpha is not None:
            # out = alpha.view(-1, self.heads, 1) * out
            out = alpha.unsqueeze(-1) * out
        return out
    ####forward是对__forward__方法的封装，传入了输入的节点特征和超图，返回超图卷积结果
    def forward(self, x, hyperedge_index, adj, hyperedge_weight=None):
        # x = torch.matmul(x, self.weight)
        # hyperedge_weight=data.edge_attr
        alpha = None

        x1=x.transpose(0,1)
        # print(hyperedge_index.shape)
        # print(hyperedge_index[0].shape)
        x_i = torch.index_select(x1, dim=0, index=hyperedge_index[0])
        edges = hyperedge_index[1]
        adj = torch.tensor(adj).to(x.device)
        # x_k=torch.index_select(x1, dim=0, index=hyperedge_index[1])
        # unique_edgesk, inverse_indicesk = torch.unique(x_k, return_inverse=True,dim=0)
        ###非重复聚合
        unique_edges, inverse_indices = torch.unique(edges, return_inverse=True)
        aggregated_values = torch.stack([torch.sum(x_i[edges == edge],dim=0) for edge in unique_edges]).transpose(0,1)
        edge_query=self.W_query(aggregated_values)
        edge_key=self.W_key(aggregated_values)
        edge_value=self.W_value(aggregated_values)
        qk=torch.matmul(edge_query,edge_key.transpose(1,2))/(self.in_channels**0.5)
        hyedge=qk-(1-adj)*500
        soft_result=torch.softmax(hyedge,dim=-1).float()
        # edge_result=soft_result*edge_value
        edge_result=torch.matmul(soft_result,edge_value).transpose(0,1)
        x_j = torch.index_select(edge_result, dim=0, index=hyperedge_index[1])
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, hyperedge_index[0], num_nodes=x1.size(0))  # [1008,1]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        D = degree(hyperedge_index[0], x1.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges / 2), x.dtype)
        B[B == float("inf")] = 0
        # D = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
        # dim=0, dim_size=num_nodes, reduce='sum')
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x1, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        out = out.transpose(0, 1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.d_model, self.out_channels)