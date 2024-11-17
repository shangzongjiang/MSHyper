from torch.functional import align_tensors
import torch.nn as nn

from torch.nn.modules.linear import Linear
from .SubLayers import MultiHeadAttention, PositionwiseFeedForward
import torch
from .embed import DataEmbedding, CustomEmbedding
import math
import numpy as np
import torch.nn.functional as F




def get_mask(input_size, window_size, inner_size, khop, device):
    """Get the attention mask of HyperGraphConv"""
    # Get the size of all layers
    # window_size=[4,4,4]
    all_size = []
    all_size.append(input_size)
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)

    ###尺度内原始连接intra_ori
    j = 0
    intra_all = []
    num_all = []
    # inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            if (i + 1) % inner_size == 0 or (i + 1 == start + all_size[layer_idx]):
                left_side = max(i - inner_size+1, start)
                right_side = min(i + 1, start + all_size[layer_idx])
                num = list(range(left_side, right_side))
                num_all += num
                intra_edge = list(np.repeat(j, len(num)))
                intra_all += intra_edge
                j += 1
    intra_ori = np.vstack((num_all, intra_all))
    intra_oir = torch.tensor(intra_ori, dtype=torch.long)
    seq_length = sum(all_size)  ###[177]

    ###尺度内跳跃连接
    # s = 2
    # h = 4
    a = []
    number_edge = []
    k = j
    for i in range(1,int(seq_length/inner_size)):
        base = (i - 1) // khop
        horizon = khop * inner_size
        bias = (i - 1) % khop + 1
        result = base * horizon + bias
        for j in range(inner_size):
            # num=list(range)
            a.append(result)
            # print(result, end=" ")
            result = result + khop
        edge_index = list(np.repeat(k, inner_size))
        number_edge += edge_index
        k += 1
    intra_resultsk = np.vstack((a, number_edge))
    intra_resultsk = torch.tensor(intra_resultsk, dtype=torch.long)
    final_intra = np.hstack((intra_ori, intra_resultsk))
    mask_intra = get_intra_mask(final_intra[1])
    final_intra = torch.tensor(final_intra, dtype=torch.long)

    ###尺度间原始连接
    j = k
    inter_all = []
    num_all = []
    # inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            if (i + 1) % inner_size == 0 or (i + 1 == start + all_size[layer_idx]):
                left_side = max(i - inner_size+1, start)
                right_side = min(i + 1, start + all_size[layer_idx])
                num = list(range(left_side, right_side))
                if layer_idx == 0:
                    scale_node = math.ceil(left_side / window_size[layer_idx - 1]) + sum(all_size[:layer_idx + 1])
                else:
                    scale_node = math.ceil((left_side - sum(all_size[:layer_idx])) / window_size[layer_idx - 1]) + sum(
                        all_size[:layer_idx + 1])
                scale_node = min(scale_node, sum(all_size) - 1)
                num_all += num
                num_all.append(scale_node)
                inter_edge = list(np.repeat(j, len(num) + 1))
                inter_all += inter_edge
                j += 1
    inter_ori = np.vstack((num_all, inter_all))
    inter_oir = torch.tensor(inter_ori, dtype=torch.long)

    ###尺度间跳跃连接
    # j=0
    inter_all = []
    num_all = []
    # inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):

            base = (i - 1) // khop
            horizon = khop * inner_size
            bias = (i - 1) % khop + 1
            result = base * horizon + bias
            for jj in range(inner_size):
                # num=list(range)
                a.append(result)
                # print(result, end=" ")
                result = result + khop
            if (i + 1) % inner_size == 0 or (i + 1 == start + all_size[layer_idx]):
                left_side = max(i - inner_size, start)
                right_side = min(i + 1, start + all_size[layer_idx])
                num = list(range(left_side, right_side))
                if layer_idx == 0:
                    scale_node = math.ceil(left_side / window_size[layer_idx - 1]) + sum(all_size[:layer_idx + 1])
                else:
                    scale_node = math.ceil((left_side - sum(all_size[:layer_idx])) / window_size[layer_idx - 1]) + sum(
                        all_size[:layer_idx + 1])
                scale_node = min(scale_node, sum(all_size) - 1)
                num_all += num
                num_all.append(scale_node)
                inter_edge = list(np.repeat(j, len(num) + 1))
                inter_all += inter_edge
                j += 1
    inter_sk = np.vstack((num_all, inter_all))
    inter_sk = torch.tensor(inter_sk, dtype=torch.long)
    final_inter = np.hstack((inter_ori, inter_sk))
    mask_inter = get_inter_mix_mask(final_inter[0], final_inter[1])
    concat_ite_ita = np.vstack((np.hstack((mask_intra, np.zeros((mask_intra.shape[0], mask_inter.shape[1])))),
                                np.hstack((np.zeros((mask_inter.shape[0], mask_intra.shape[1])), mask_inter))))
    final_inter = torch.tensor(final_inter, dtype=torch.long)

    # all_size = [169, 42, 10, 2]
    # h = 3
    # k = 1
    # s = 1  # 代表间隔
    # a = []  # 代表收集的数据
    # for layer_idx in range(len(all_size)):
    #     start = sum(all_size[:layer_idx])
    #     print(start)
    #     print(start + all_size[layer_idx] - 1)
    #     print("---------------------")
    #     for i in range(1, all_size[layer_idx] + 1):
    #         base = (i - 1) // (s + 1)
    #         horizon = (s + 1) * h
    #         bias = (i - 1) % (s + 1) + 1
    #         result = base * horizon + bias
    #         result = result + start
    #         # result-1+ (h-1)*(s+1) 计算出最后一个编号 如果这个编号不在本层 直接brek跳出
    #         if result - 1 + (h - 1) * (s + 1) > (start + all_size[layer_idx] - 1):
    #             # print(result-1+ (h-1)*(s+1),"qweqwe")
    #             break
    #         for j in range(h):
    #             # num=list(range)
    #             a.append(result)
    #             print(result - 1, end=" ")
    #             result = result + s + 1
    #         print()

    # print(intra_resultsk)

    h = 4
    mixnum_all = []
    mix_all = []
    # j=0
    for i in range(1, all_size[0] + 1):
        # print(i)
        if (i % h == 0 and i > 0) or (i == (all_size[0])):
            left_side = i - h
            right_side = min(i, all_size[0] + 1)
            num = list(range(left_side, right_side))
            mixnum_all += num
            node_scale1 = math.ceil(left_side / sum(window_size[:1])) + sum(all_size[:1]) + 1
            mixnum_all.append(node_scale1)
            node_scale2 = math.ceil(left_side / (window_size[0] * window_size[1])) + sum(all_size[:2])
            node_scale2 = min(node_scale2, sum(all_size[:3]))
            mixnum_all.append(node_scale2)
            node_scale3 = math.ceil(left_side / (window_size[0] * window_size[1] * window_size[2])) + sum(all_size[:3])
            node_scale3 = min(node_scale3, sum(all_size[:4]), sum(all_size[:4]) - 1)
            mixnum_all.append(node_scale3)
            mix_edge = list(np.repeat(j, len(num) + 3))
            # for s in range(len(window_size)):
            #     node_scale=math.ceil(right_side/window_size[s])
            mix_all += mix_edge
            j += 1
    # mix_result = np.vstack((mixnum_all, mix_all))
    mix_result = np.vstack((mixnum_all, mix_all))
    mask_mix = get_inter_mix_mask(mix_result[0], mix_result[1])
    concat_all = np.vstack((np.hstack((concat_ite_ita, np.zeros((concat_ite_ita.shape[0], mask_mix.shape[1])))),
                            np.hstack((np.zeros((mask_mix.shape[0], concat_ite_ita.shape[1])), mask_mix))))
    mix_result = torch.tensor(mix_result, dtype=torch.long)
    """
    ###备份
    for i in range(1,all_size[0]+1):
        # print(i)
        if (i%h==0 and i>0) or (i==(all_size[0])):
            left_side=i-h
            right_side=min(i,all_size[0]+1)
            num=list(range(left_side,right_side))
            mixnum_all+=num
            node_scale1=math.ceil(left_side/sum(window_size[:1]))+sum(all_size[:1])+1
            mixnum_all.append(node_scale1)
            node_scale2=math.ceil(left_side/(window_size[0]*window_size[1]))+sum(all_size[:2])
            node_scale2=min(node_scale2,sum(all_size[:3]))
            mixnum_all.append(node_scale2)
            node_scale3 = math.ceil(left_side / (window_size[0]*window_size[1]*window_size[2])) + sum(all_size[:3])
            node_scale3=min(node_scale3,sum(all_size[:4]),sum(all_size[:4])-1)
            mixnum_all.append(node_scale3)
            mix_edge = list(np.repeat(j, len(num)+3))
            # for s in range(len(window_size)):
            #     node_scale=math.ceil(right_side/window_size[s])
            mix_all += mix_edge
            j += 1
    # mix_result = np.vstack((mixnum_all, mix_all))
    mix_result_all = np.vstack((mixnum_all, mix_all))
    # concat_all = np.vstack((np.hstack((concat_ite_ita, np.zeros((concat_ite_ita.shape[0], mix_result_all.shape[1])))),
    #                             np.hstack((np.zeros((mix_result_all.shape[0], concat_ite_ita.shape[1])), mix_result_all))))
    mix_result = torch.tensor(mix_result_all, dtype=torch.long)
    """

    # edge_index=np.hstack((edge_index,edge_inter))
    # edge_index = np.hstack((edge_index, intra_result6))
    # edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = np.hstack((final_intra, final_inter))
    edge_index = np.hstack((edge_index, mix_result))
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return final_intra, all_size,mask_intra
    # edge_index_test
    # return intra_result6, all_size
    # return intra_resultsk, all_size,mask_intra




def get_intra_mask(intra_mask):
    length=max(intra_mask)
    mask = np.zeros((length+1, length+1))
    for i in range(length+1):
        for j in range(length+1):
            if 0 <= i - j <= 1:
                mask[i, j] = 1
    return mask

def get_inter_mix_mask(nodes,edges):
    num_nodes = max(nodes) + 1  # 获取节点的数量
    num_edges = max(edges) + 1  # 获取超边的数量

    mask = np.zeros((num_edges, num_edges))  # 创建初始的mask矩阵

    for i in range(len(edges)):
        for j in range(len(edges)):
            if nodes[i] == nodes[j]:
                mask[edges[i], edges[j]] = 1  # 根据条件设置mask值
                mask[edges[j], edges[i]] = 1  # 由于是无向图，需要对称设置

    return mask

def refer_points(all_sizes, window_size, device):
    """Gather features from PAM's pyramid sequences"""
    input_size = all_sizes[0]###all_size=[169,42,10,2]
    indexes = torch.zeros(input_size, len(all_sizes), device=device)###index=[169,4]

    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(all_sizes)):
            start = sum(all_sizes[:j])
            inner_layer_idx = former_index - (start - all_sizes[j - 1])
            former_index = start + min(inner_layer_idx // window_size[j - 1], all_sizes[j] - 1)
            indexes[i][j] = former_index

    indexes = indexes.unsqueeze(0).unsqueeze(3)###unsqueeze(0)在0这个维度增加一维

    return indexes.long()


def get_subsequent_mask(input_size, window_size, predict_step, truncate):
    """Get causal attention mask for decoder."""
    if truncate:
        mask = torch.zeros(predict_step, input_size + predict_step)
        for i in range(predict_step):
            mask[i][:input_size+i+1] = 1
        mask = (1 - mask).bool().unsqueeze(0)
    else:
        all_size = []
        all_size.append(input_size)
        for i in range(len(window_size)):
            layer_size = math.floor(all_size[i] / window_size[i])
            all_size.append(layer_size)
        all_size = sum(all_size)
        mask = torch.zeros(predict_step, all_size + predict_step)
        for i in range(predict_step):
            mask[i][:all_size+i+1] = 1
        mask = (1 - mask).bool().unsqueeze(0)

    return mask


def get_q_k(input_size, window_size, stride, device):
    """
    Get the index of the key that a given query needs to attend to.
    """
    second_length = input_size // stride
    second_last = input_size - (second_length - 1) * stride
    third_start = input_size + second_length
    third_length = second_length // stride
    third_last = second_length - (third_length - 1) * stride
    max_attn = max(second_last, third_last)
    fourth_start = third_start + third_length
    fourth_length = third_length // stride
    full_length = fourth_start + fourth_length
    fourth_last = third_length - (fourth_length - 1) * stride
    max_attn = max(third_last, fourth_last)

    max_attn += window_size + 1
    mask = torch.zeros(full_length, max_attn, dtype=torch.int32, device=device) - 1

    for i in range(input_size):
        mask[i, 0:window_size] = i + torch.arange(window_size) - window_size // 2
        mask[i, mask[i] > input_size - 1] = -1

        mask[i, -1] = i // stride + input_size
        mask[i][mask[i] > third_start - 1] = third_start - 1
    for i in range(second_length):
        mask[input_size+i, 0:window_size] = input_size + i + torch.arange(window_size) - window_size // 2
        mask[input_size+i, mask[input_size+i] < input_size] = -1
        mask[input_size+i, mask[input_size+i] > third_start - 1] = -1

        if i < second_length - 1:
            mask[input_size+i, window_size:(window_size+stride)] = torch.arange(stride) + i * stride
        else:
            mask[input_size+i, window_size:(window_size+second_last)] = torch.arange(second_last) + i * stride

        mask[input_size+i, -1] = i // stride + third_start
        mask[input_size+i, mask[input_size+i] > fourth_start - 1] = fourth_start - 1
    for i in range(third_length):
        mask[third_start+i, 0:window_size] = third_start + i + torch.arange(window_size) - window_size // 2
        mask[third_start+i, mask[third_start+i] < third_start] = -1
        mask[third_start+i, mask[third_start+i] > fourth_start - 1] = -1

        if i < third_length - 1:
            mask[third_start+i, window_size:(window_size+stride)] = input_size + torch.arange(stride) + i * stride
        else:
            mask[third_start+i, window_size:(window_size+third_last)] = input_size + torch.arange(third_last) + i * stride

        mask[third_start+i, -1] = i // stride + fourth_start
        mask[third_start+i, mask[third_start+i] > full_length - 1] = full_length - 1
    for i in range(fourth_length):
        mask[fourth_start+i, 0:window_size] = fourth_start + i + torch.arange(window_size) - window_size // 2
        mask[fourth_start+i, mask[fourth_start+i] < fourth_start] = -1
        mask[fourth_start+i, mask[fourth_start+i] > full_length - 1] = -1

        if i < fourth_length - 1:
            mask[fourth_start+i, window_size:(window_size+stride)] = third_start + torch.arange(stride) + i * stride
        else:
            mask[fourth_start+i, window_size:(window_size+fourth_last)] = third_start + torch.arange(fourth_last) + i * stride

    return mask


def get_k_q(q_k_mask):
    """
    Get the index of the query that can attend to the given key.
    """
    k_q_mask = q_k_mask.clone()
    for i in range(len(q_k_mask)):
        for j in range(len(q_k_mask[0])):
            if q_k_mask[i, j] >= 0:
                k_q_mask[i, j] = torch.where(q_k_mask[q_k_mask[i, j]] ==i )[0]

    return k_q_mask


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True, use_tvm=False, q_k_mask=None, k_q_mask=None):
        super(EncoderLayer, self).__init__()
        self.use_tvm = use_tvm
        if use_tvm:
            from .PAM_TVM import PyramidalAttention
            self.slf_attn = PyramidalAttention(n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before, q_k_mask=q_k_mask, k_q_mask=k_q_mask)
            ###n_head=6, d_model=512, d_k=128, d_v=128
        else:
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, slf_attn_mask=None):
        if self.use_tvm:
            enc_output = self.slf_attn(enc_input)
            enc_slf_attn = None
        else:
            enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)##enc_input[32,223,512] mask[32,223,223]

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, Q, K, V, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            Q, K, V, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=window_size,
                                  stride=window_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)####input_x[32,128,42]
        x = self.norm(x)
        x = self.activation(x)
        return x


class Conv_Constructnew(nn.Module):
    """Convolution CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(Conv_Construct, self).__init__()
        # if not isinstance(window_size, list):
        #     self.conv_layers = nn.ModuleList([
        #         ConvLayer(d_model, window_size),
        #         ConvLayer(d_model, window_size),
        #         ConvLayer(d_model, window_size)
        #         ])
        # else:
        #     self.conv_layers = nn.ModuleList([
        #         ConvLayer(d_model, window_size[0]),
        #         ConvLayer(d_model, window_size[1]),
        #         ConvLayer(d_model, window_size[2])
        #         ])
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_model, window_size),
                ConvLayer(d_model, window_size)
                # ConvLayer(d_model, window_size)
            ])
        else:
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_model, window_size[0]),
                ConvLayer(d_model, window_size[1])
                # ConvLayer(d_model, window_size[2])
                ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.permute(0, 2, 1)
        all_inputs.append(enc_input)

        for i in range(len(self.conv_layers)):
            enc_input = self.conv_layers[i](enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs
class Conv_Construct(nn.Module):
    """Convolution CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(Conv_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_model, window_size),
                ConvLayer(d_model, window_size),
                ConvLayer(d_model, window_size)
                ])
        else:
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_model, window_size[0]),
                ConvLayer(d_model, window_size[1]),
                ConvLayer(d_model, window_size[2])
                ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.permute(0, 2, 1)
        all_inputs.append(enc_input)

        for i in range(len(self.conv_layers)):
            enc_input = self.conv_layers[i](enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs

class Bottleneck_Construct(nn.Module):
    """Bottleneck convolution CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(Bottleneck_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size)
                ])
        else:
            self.conv_layers = []
            for i in range(len(window_size)):
                self.conv_layers.append(ConvLayer(d_inner, window_size[i]))
            self.conv_layers = nn.ModuleList(self.conv_layers)
        self.up = Linear(d_inner, d_model)####d_inner128 d_model=512
        self.down = Linear(d_model, d_inner)####d_model=512 d_inner128
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):####[32,169,512]

        temp_input = self.down(enc_input).permute(0, 2, 1)####先下采样，变为[32,169,128],再交换第1维度和第二维度-->[32,128,169]
        all_inputs = []
        ####对169个节点进行卷积
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)####第一次[32,128,42]第二次[32,128,10]第三次[32,128,2]
            all_inputs.append(temp_input)

        # print(all_inputs[1].shape())
        # hour_trend=self.up(all_inputs[0].transpose(1, 2))
        # day_trend=self.up(all_inputs[1].transpose(1, 2))
        # week_trend=self.up(all_inputs[2].transpose(1, 2))
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)####[32,54,128]
        all_inputs = self.up(all_inputs)####[32,54,512]
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)####enc_input[32,169,512]all_input=[32,223,512]

        all_inputs = self.norm(all_inputs)####[32,223,512]
        # hour_trend = self.norm(hour_trend)
        # day_trend = self.norm(day_trend)
        # week_trend = self.norm(week_trend)

        return all_inputs



class MaxPooling_Construct(nn.Module):
    """Max pooling CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(MaxPooling_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.pooling_layers = nn.ModuleList([
                nn.MaxPool1d(kernel_size=window_size),
                nn.MaxPool1d(kernel_size=window_size),
                nn.MaxPool1d(kernel_size=window_size)
                ])
        else:
            self.pooling_layers = nn.ModuleList([
                nn.MaxPool1d(kernel_size=window_size[0]),
                nn.MaxPool1d(kernel_size=window_size[1]),
                nn.MaxPool1d(kernel_size=window_size[2])
                ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        all_inputs.append(enc_input)

        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs


class AvgPooling_Construct(nn.Module):
    """Average pooling CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(AvgPooling_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.pooling_layers = nn.ModuleList([
                nn.AvgPool1d(kernel_size=window_size),
                nn.AvgPool1d(kernel_size=window_size),
                nn.AvgPool1d(kernel_size=window_size)
                ])
        else:
            self.pooling_layers = nn.ModuleList([
                nn.AvgPool1d(kernel_size=window_size[0]),
                nn.AvgPool1d(kernel_size=window_size[1]),
                nn.AvgPool1d(kernel_size=window_size[2])
                ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        all_inputs.append(enc_input)

        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs


class Predictor(nn.Module):

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data):
        out = self.linear(data)
        out = out
        return out


class Decoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, opt, mask):
        super().__init__()

        self.model_type = opt.model
        self.mask = mask

        self.layers = nn.ModuleList([
            DecoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout, \
                normalize_before=False),
            DecoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout, \
                normalize_before=False)
            ])

        if opt.embed_type == 'CustomEmbedding':
            self.dec_embedding = CustomEmbedding(opt.enc_in, opt.d_model, opt.covariate_size, opt.seq_num, opt.dropout)
        else:
            self.dec_embedding = DataEmbedding(opt.enc_in, opt.d_model, opt.dropout)

    def forward(self, x_dec, x_mark_dec, refer):
        dec_enc = self.dec_embedding(x_dec, x_mark_dec)

        dec_enc, _ = self.layers[0](dec_enc, refer, refer)
        refer_enc = torch.cat([refer, dec_enc], dim=1)
        mask = self.mask.repeat(len(dec_enc), 1, 1).to(dec_enc.device)
        dec_enc, _ = self.layers[1](dec_enc, refer_enc, refer_enc, slf_attn_mask=mask)

        return dec_enc

