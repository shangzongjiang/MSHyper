import torch
import torch.nn as nn
from .Layers import EncoderLayer, Decoder, Predictor
from .Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from .Layers import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
from .embed import DataEmbedding, CustomEmbedding,DataEmbedding_new
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.data import data as D
from torch.nn import Linear
import numpy as np
import torch.nn.functional as F
import torch_scatter
from math import sqrt



class Model(nn.Module):
    def __init__(self,opt):
        super(Model, self).__init__()
        self.predict_step = opt.predict_step###预测长度
        self.d_model = opt.d_model###维度
        self.model_type = opt.model###模型类型
        self.window_size = opt.window_size
        self.truncate = opt.truncate
        self.CSCM=opt.CSCM###选择构建多尺度方式，卷积池化等
        self.mask, self.all_size = get_mask(opt.input_size + 1, opt.window_size, opt.inner_size, opt.device)###基于先验的超边构建
        self.indexes = refer_points(self.all_size, opt.window_size, opt.device)
        # self.conv1 = GCNConv(dataset_cora.num_node_features, 168)
        # self.conv2 = GCNConv(168, dataset_cora.num_classes)
        self.conv1 = HypergraphConv(512, 512)###定义下面的超图卷积方式
        # self.conv2 = HypergraphConv(512, 512)####原始[32,512]
        # self.conv3 = HypergraphConv(512, 512)###原始[512,512]
        # self.conv1 = GCNConv(1, 16)
        # self.conv2 = GCNConv(16, 2)
        self.enc_embedding = DataEmbedding(opt.enc_in, opt.d_model, opt.dropout)####数据嵌入方式
        self.conv_layers = eval(opt.CSCM)(opt.d_model, opt.window_size, opt.d_bottleneck)

        # self.predict_step = opt.predict_step
        # self.d_model = opt.d_model
        self.input_size = opt.input_size
        # self.decoder_type = opt.decoder
        self.channels = opt.enc_in
        self.predictor = Predictor(4 * opt.d_model, opt.predict_step * opt.enc_in)
        # self.predictor = Predictor(opt.d_model, opt.predict_step * opt.enc_in)

        self.indexes = refer_points(self.all_size, opt.window_size, opt.device)

        # self.trans = nn.Linear(in_features = 223*512,out_features = self.predict_step)###原始为223*16
        self.convtra=nn.Linear(169,169)
        # self.convtra1=nn.Linear(512,7)

        # self.encoder = LLA(opt)


    def forward(self, x_enc,x_mark_enc,x_dec, x_mark_dec, pretrain):
        # enc_output = self.encoder(x_enc, x_mark_enc)
        #####input_normalization

        mean_enc=x_enc.mean(1,keepdim=True).detach()
        x_enc=x_enc - mean_enc
        std_enc=torch.sqrt(torch.var(x_enc,dim=1,keepdim=True,unbiased=False)+1e-5).detach()
        x_enc=x_enc / std_enc



        seq_enc = self.enc_embedding(x_enc, x_mark_enc)###嵌入方式选择 在embed.py里
        mask = self.mask.repeat(len(seq_enc), 1, 1) #[32,2,223]
        mask = torch.tensor(mask).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc).to(x_enc.device)

        # if self.CSCM=="AvgPooling_Construct" or self.CSCM=="Conv_Construct":####卷积方式设置
        #     seq_enc = self.conv_layers(seq_enc).to(x_enc.device)  # [32,223,512]
        # else:
        #     seq_enc,hour_trend,day_trend,week_trend, all_trend = self.conv_layers(seq_enc)#[32,223,512]

        seq_enc = torch.tensor(seq_enc, dtype=torch.float).to(x_enc.device)
        x = self.conv1(seq_enc, mask)###x[32,223,512]####超图卷积
        """
        x=self.convtra1(x)
        x=x.permute(0,2,1)
        x=self.convtra(x)
        pred=x.permute(0,2,1)
        """
        # x=x+all_trend
        # x=x+seq_enc
        # x = self.conv2(x,mask)
        # x=x+seq_enc
        # x=self.conv3(x,mask)
        # trend_all=torch.cat([hour_trend, day_trend,week_trend,x], dim=1)
        # x=self.convtra(x)
        # x = F.relu(x)#####x=[32,223,512]####不用relu和dropout效果会好一些
        # x = F.dropout(x, training=self.training)

        ####以下代码为自己实现的FC
        """
        x=x.contiguous().view(x.size(0),-1)####x=[32,223,16]--[32,223x16]###因为地址不连续，不加contiguous可能会报错
        x=self.trans(x)
        pred=x.view(x.size(0),self.predict_step,-1)
        """



        #####以下代码为原始代码，即金字塔最后一维输出

        indexes = self.indexes.repeat(x.size(0), 1, 1, x.size(2)).to(x.device)  ##seq_enc.size(0)(1)(2)=32,223,512  indexes=[32,169,4,512]
        indexes = indexes.view(x.size(0), -1, x.size(2))  ###index=[32,169x4,512]=[32,676,512]
        all_enc = torch.gather(x, 1, indexes)  # seq_enc=[32,223,512] index=[32,676,512] all_enc=[32,676,512]
        seq_enc = all_enc.view(x.size(0), self.all_size[0], -1)  ###seq_enc=[32,169,2048]


        seq_enc=seq_enc.permute(0,2,1)
        seq_enc=self.convtra(seq_enc)
        seq_enc=seq_enc.permute(0,2,1)

        enc_output = seq_enc[:, -1, :]#####-1只取最后一维[32,2048]
        # enc_output=x[:, 0:168, :]
        # enc_output = x[:, 0:4, :]
        # enc_output=enc_output.reshape(32,2048)
        pred = self.predictor(enc_output).view(enc_output.size(0), self.predict_step, -1)####Predictor 线性函数输入2048 输出168 prpred[32,168,1]


        ######de_normalization
        # pred=pred*std_enc +mean_enc
        # print(x)




        # #添加data测试
        # data = D.Data()
        # data.x, data.edge_index \
        #     = seq_enc, mask
        # x = self.conv1(data.x, data.edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # # x = self.conv2(x, edge_index)
        # x = F.softmax(x, dim=1)
        pred = pred * std_enc + mean_enc
        return pred

####Messsagepassing是Geometric的基本模块
class HypergraphConv_att(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_attention=True,
                 heads=4,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=False):
        super(HypergraphConv, self).__init__(aggr='add')
        self.soft=nn.Softmax(dim=0)
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
#初始化权重和偏置参数
    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)
####计算超图的度，然后根据度计算超图的归一化权重
    def __forward__(self,
                    x,
                    hyperedge_index,
                    hyperedge_weight=None,
                    alpha=None):
        # print('okk')
        # print(x.size(0))
        # bbb=x[1,:,:]
        # print(bbb)
        # print(x[2,:,:])
        aa = hyperedge_index[0]
        dd=x.size(0)
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
        ####else这条路径没有走
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

        #####B是超边的度
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
        # out = norm[edge_index_i].view(-1, 1, 1) * x_j
        ###attention 操作
        out = norm[edge_index_i].view(-1, 1, 1) * x_j.view(
            -1, self.heads, self.out_channels)
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out
        return out
    ####forward是对__forward__方法的封装，传入了输入的节点特征和超图，返回超图卷积结果
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

        # if self.use_attention:
        #     x = x.view(-1,-1, self.heads, self.out_channels)####[32,223,512]--[7136,1,512]
        #     x_i, x_j = x[hyperedge_index[0]], x[hyperedge_index[1]]#####x_i,x_j[2,223,1,512]
        #     alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)####alpha=[2,223,1]
        #     alpha = F.leaky_relu(alpha, self.negative_slope)
        #     alpha = softmax(alpha, hyperedge_index[0], x.size(0))
        #     alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        for i in range(x.size(0)):
            x_new = x[i, :, :]
            hyperedge_index_new = hyperedge_index[i, :, :]
            if self.use_attention:
                # aaa=x[i,:,:]#####[32,223,512]--[223,512]
                # x=x[i,:,:]
                # hyperedge_index=hyperedge_index[i,:,:]
                #attention 操作
                x_new = x_new.view(-1, self.heads, self.out_channels)  ####[223,512]--[223,head,512]
                # x_new = x_new.view(x_new.size(0), self.heads, -1)
                x_i, x_j = x_new[hyperedge_index_new[0]], x_new[hyperedge_index_new[1]]  #####x_i,x_j[2,223,1,512]

                ###add_start
                # x_jnew=x_j.permute(0,2,1)
                # alpha_new = torch.bmm(x_i, x_jnew)
                # alpha_new=self.soft(alpha_new)
                # alpha_new = F.dropout(alpha_new, p=self.dropout, training=self.training)
                # x_last=torch.bmm(alpha_new,x_j).sum(dim=-1)
                ###add_end

                ###origional
                alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  ####alpha=[223,head]
                alpha = F.leaky_relu(alpha, self.negative_slope)
                # alpha = F.leaky_relu(x_last, self.negative_slope)####x_last的时候用
                alpha = softmax(alpha, hyperedge_index_new[0], num_nodes=x_new.size(0))
                # alpha = softmax(alpha, hyperedge_index_new[0], x_new.size(0))

                alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            # out_batch = self.__forward__(x[i,:,:], hyperedge_index[i,:,:], hyperedge_weight, alpha)
            out_batch = self.__forward__(x_new, hyperedge_index_new, hyperedge_weight, alpha)###out_batch=[223,1024]
            out_batch = out_batch.view(-1,1, self.heads * self.out_channels)  ####out=[224,32]
            # out_batch = out_batch.view(-1, 1, self.out_channels)
            out_batch = self.fc(out_batch)
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
class HypergraphConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_attention=True,
                 heads=8,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.1,
                 bias=False):
        super(HypergraphConv, self).__init__(aggr='add')
        self.soft=nn.Softmax(dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        # self.fc = nn.Linear(self.out_channels * heads, self.out_channels)

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
                    hyperedge_index,
                    hyperedge_weight=None,
                    alpha=None):
        # print('okk')
        # print(x.size(0))
        # bbb=x[1,:,:]
        # print(bbb)
        # print(x[2,:,:])
        aa = hyperedge_index[0]
        dd=x.size(0)
        # print(hyperedge_index[0][0].size(0))
        # bbb=hyperedge_index##[32,2,223]
        # aaa=hyperedge_index[-1,:,:]##[2,223]
        # cc=x    ###[32,223,512]
        # print(hyperedge_index[0][1])
        # print(x.szie(0))
        # print("okkk")

        # hyperedge_index=hyperedge_index[-1,:,:]
        # x=x[-1,:,:]
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
    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        x = torch.matmul(x, self.weight)
        # hyperedge_weight=data.edge_attr
        alpha = None

        # if self.use_attention:
        #     x = x.view(-1,-1, self.heads, self.out_channels)####[32,223,512]--[7136,1,512]
        #     x_i, x_j = x[hyperedge_index[0]], x[hyperedge_index[1]]#####x_i,x_j[2,223,1,512]
        #     alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)####alpha=[2,223,1]
        #     alpha = F.leaky_relu(alpha, self.negative_slope)
        #     alpha = softmax(alpha, hyperedge_index[0], x.size(0))
        #     alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        for i in range(x.size(0)):
            x_new = x[i, :, :]
            #####hyperedge_index_new[0]是节点索引，hyperedge_index_new[1]是超边索引
            hyperedge_index_new = hyperedge_index[i, :, :]

            if self.use_attention:
                # aaa=x[i,:,:]#####[32,223,512]--[223,512]
                # x=x[i,:,:]
                # hyperedge_index=hyperedge_index[i,:,:]
                # print(x_new.size(0))
                # x_new = x_new.view(-1, self.heads, self.out_channels)####[223,512]--[223,head,512]
                x_new = x_new.view(x_new.size(0), self.heads, -1) ####[223,512]--[223,head,512]
                ####把对应位置的值拿出来
                x_i, x_j = x_new[hyperedge_index_new[0]], x_new[hyperedge_index_new[1]]  #####x_i,x_j[2,223,1,512]
                """
                ####生成超边特征
                hyperedge_index_node=x_i.view(x_i.size(0),-1)
                hyperedge_index_edge=hyperedge_index_new[1]
                superedges={}
                hyperedge_index_edge=hyperedge_index_edge.cpu()
                hyperedge_index_edge=list(hyperedge_index_edge.numpy())
                for j in range(len(hyperedge_index_node)):
                    if hyperedge_index_edge[j] not in superedges:
                        superedges[hyperedge_index_edge[j]] = hyperedge_index_node[j]
                    else:
                        superedges[hyperedge_index_edge[j]] =superedges[hyperedge_index_edge[j]] + hyperedge_index_node[j]
                # superedges = torch.tensor(superedges, dtype=torch.long)
                # print(type(hyperedge_index_edge))
                # hyperedge_index_edge=hyperedge_index_edge.cpu()
                # hyperedge_index_edge=hyperedge_index_edge.numpy()
                # superedges=superedges.cpu()
                # print(type(hyperedge_index_edge))
                # print(type(superedges))

                for j in range(len(hyperedge_index_edge)):
                    # print
                    # print(type(hyperedge_index_edge))
                    hyperedge_index_edge[j] = superedges[hyperedge_index_edge[j]]
                x_j=torch.stack(hyperedge_index_edge,dim=0).view(x_i.size(0),self.heads, -1)
                """

                """
                ###add_start
                x_jnew=x_j.permute(0,2,1)
                alpha_new = torch.bmm(x_i, x_jnew)
                alpha_new=self.soft(alpha_new)
                alpha_new = F.dropout(alpha_new, p=self.dropout, training=self.training)
                x_last=torch.bmm(alpha_new,x_j).sum(dim=-1)
                ###add_end
                """

                ###origional
                alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  ####alpha=[223,head]
                alpha = F.leaky_relu(alpha, self.negative_slope)
                alpha = softmax(alpha, hyperedge_index_new[0], num_nodes=x_new.size(0))
                # alpha = softmax(alpha, hyperedge_index_new[0], x_new.size(0))

                alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            # out_batch = self.__forward__(x[i,:,:], hyperedge_index[i,:,:], hyperedge_weight, alpha)
            out_batch = self.__forward__(x_new, hyperedge_index_new, hyperedge_weight, alpha)###out_batch=[223,1024]
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




