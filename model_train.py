import sys,os
from statistics import mode
import torch
sys.path.append(os.getcwd())
import argparse
from rumorDc.BiGCN_master.Process.process import *
import torch as th
from torch_scatter  import scatter_mean
import torch.nn.functional as F
import numpy as np
from rumorDc.BiGCN_master.tools.earlystopping3class import EarlyStopping
# from rumorDc.ACLR_master.ACLR4RUMOR_NAACL2022_main.tools.earlystopping2class import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from rumorDc.BiGCN_master.Process.rand5fold import *
from rumorDc.BiGCN_master.tools.evaluate import *
from torch_geometric.nn import GCNConv
import copy
from rumorDc.BiGCN_master.model.TransformerBlock  import MultiheadAttention
import torch.nn.init as init
import pickle
from collections import OrderedDict
from torch.nn import BatchNorm1d
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from rumorDc.BiGCN_master.model.Twitter.COMET import COMET
from rumorDc.ts2vec.models.dilated_conv import DilatedConvEncoder
from rumorDc.ts2vec.models.encoder import *
from rumorDc.CDGTN.utils import length_to_mask
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GATConv
#加载训练好的模型参数
class Source_COMET(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(Source_COMET, self).__init__()
        self.COMET = COMET()  # 常识推断器
    def forward(self, data):
        triple_ids, triple_mask = data.triple_ids, data.triple_mask
        triple_ids=torch.stack([torch.tensor(item) for item in triple_ids]).to(device)
        triple_mask = torch.stack([torch.tensor(item) for item in triple_mask]).to(device)
        logits=self.COMET(triple_ids,triple_mask).view(-1,768)
        logits=F.relu(logits)
        logits= F.dropout(logits, training=self.training)
        return logits


class TSEncoder(th.nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=768, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = th.nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = th.nn.Dropout(p=0.1)
    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1) #得到一个布尔类型的数组，判断行中有无缺失值;
        x[~nan_mask] = 0 #将没有缺失值的行设置为0
        x = self.input_fc(x)  # B x T x Ch
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        #mask &= nan_mask
        x[~mask] = 0
        # conv encoder
        x = x.transpose(0, 1)  # B x Ch x T
        #x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(0, 1)  # B x T x Co
        return x


class Attention(th.nn.Module):
    def __init__(self, in_features, hidden_size):
        super(Attention, self).__init__()
        self.linear1 = th.nn.Linear(in_features*2, hidden_size)
        self.linear2 = th.nn.Linear(hidden_size, 1)
        self.activation = th.nn.ReLU()
        self.dropout = th.nn.Dropout(0.5)
        self.reset_parameters()
    def reset_parameters(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)
    def forward(self, K, V, mask = None):
        '''
        :param K: (batch_size, d)
        :param V: (batch_size, hist_len, d)
        :return: (batch_size, d)
        '''
        #K = K.unsqueeze(dim=1).expand(V.size())
        #K= K.unsqueeze(0).expand(V.size(0), -1)
        fusion = th.cat([K, V], dim=-1)
        fc1 = self.activation(self.linear1(fusion))
        score = self.linear2(fc1)
        if mask is not None:
            mask = mask.unsqueeze(dim=-1)
            score = score.masked_fill(mask, -2 ** 32 + 1)
        alpha = F.softmax(score, dim=1)
        alpha = self.dropout(alpha)
        #att = (alpha * V).sum(dim=1)
        att = (alpha * V)
        return att

class TDrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)
        # self.conv3=GCNConv(hid_feats+out_feats,hid_feats)
        # self.conv1 = TransformerConv(hid_feats, hid_feats // 8, heads=8)
        # self.conv2 = TransformerConv(hid_feats*2, out_feats // 8, heads=8)
        # self.conv1 = GATConv(in_feats, 128, heads=6)
        # self.conv2 = GATConv(in_feats*2, 8, heads=8)
        self.att_i = th.nn.Linear(hid_feats*24, hid_feats)
        self.att_s = th.nn.Linear(hid_feats, 1)
        self.att_inder = th.nn.Linear(hid_feats * 2, hid_feats)
        self.mh_attention = MultiheadAttention(input_size=768, output_size=768)
        self.COMET = Source_COMET(in_feats, hid_feats, out_feats)
        #self.attention = Attention(768, 768)  # 每项*1 初始化
        self.layer_list = OrderedDict()
        self.num_features_list = [768 * r for r in [1]]
        # self.COMET=COMET()#常识推断器
        #self.gnn1 = TransformerConv(in_feats, in_feats // 2, heads=2).to(device)
        self.gnn1 = GATConv(in_feats, 128, heads=6)
        # self.TSEncoder=TSEncoder(input_dims=768, output_dims=768)
        def creat_network(self, hid_feats=768):  # 添加了一个网络,或许可以从这个网路入手来加入立场因素，进而决定相似度的计算来达到共同学习的目的。
            layer_list = OrderedDict()  # 创建一个字典
            for l in range(len(self.num_features_list)):
                layer_list['conv{}'.format(l)] = th.nn.Conv1d(  # 创建卷积层字典，而且是根据特征列表的长度
                    in_channels=hid_feats,
                    out_channels=hid_feats,
                    kernel_size=1,
                    bias=False)

                layer_list['norm{}'.format(l)] = th.nn.BatchNorm1d(
                    num_features=hid_feats)  # 创建字典归一化函数和激活函数
                layer_list['relu{}'.format(l)] = th.nn.LeakyReLU()

            layer_list['conv_out'] = th.nn.Conv1d(in_channels=hid_feats,
                                                  out_channels=hid_feats,
                                                  kernel_size=1
                                                  )  # 卷积层，最后一层用做输出
            return layer_list

        self.sim_network = th.nn.Sequential(
            creat_network(self))  # 创建了一个名为“sim_val"的神经网络,计算两个节点距离的相似度问题，应该再仔细分析原理，弄明白再处理
        mod_self = self
        mod_self.num_features_list = [hid_feats]

        self.W_mean = th.nn.Sequential(creat_network(mod_self))  # 又创建了一系列的神经网络序列
        self.W_bias = th.nn.Sequential(creat_network(mod_self))
        self.B_mean = th.nn.Sequential(creat_network(mod_self))
        self.B_bias = th.nn.Sequential(creat_network(mod_self))

        self.fc1 = th.nn.Linear(hid_feats, 2, bias=False)
        self.fc2 = th.nn.Linear(hid_feats, 2, bias=False)
        self.dropout = th.nn.Dropout(0.2)
        self.eval_loss = th.nn.KLDivLoss(reduction='batchmean')
        self.bn1 = BatchNorm1d(hid_feats + in_feats)

    def local_attention_network(self, X_source, X_replies, mask):  # mask是一个掩码用于指示哪些元素需要进行注意力计算
        #可以更加关注源数据和回复数据之间的关联性，从而提高下游任务的性能。
        #X_replies=self.edge_infer(X_source,X_replies, edge_index,x_batch) #暂定为边损失的模块
        X_srouce_feat = X_source
        X_replies_feat = X_replies #这里可以进一步改成根节点和第一圈的节点的关系，而不必须是全部回复之间的关系，目的得到根节点和离他最近的节点的信息的融合以及他们之间边的路径选择。
        #这里或许可以利用多头注意力进行时间的增量运算
        # x_out = self.mh_attention(X_replies_feat, X_srouce_feat, X_replies_feat)
        X_replies_feat= self.mh_attention(X_replies_feat, X_replies_feat, X_replies_feat)
        #x_out = self.multi_head_attention(X_replies_feat, X_srouce_feat, X_replies_feat, batch)
        attention = Attention(X_srouce_feat.size(1), X_srouce_feat.size(1)) #动态注意力机制
        attention=attention.to(device)
        X_att = attention(X_srouce_feat, X_replies_feat, mask=mask) #这里这一部分是源帖子和全部回复之间的注意力分数的信息与原来的根节点帖子信息进行合并，主要得出的是根节点信息。
        X_fuse = th.cat([X_srouce_feat, X_att], dim=-1)
        linear_fuse = th.nn.Linear(X_fuse.size(1), 1)
        linear_fuse = linear_fuse.to(device)
        linear_output = linear_fuse(X_fuse.to(device))
        #alpha = th.sigmoid(linear_fuse(X_fuse.to(device)))
        alpha = th.sigmoid(linear_output)
        #X_att=X_att[:768]
        # zeros = th.zeros(len(X_srouce_feat) -len(X_att) , dtype=X_att.dtype)
        # X_att= th.cat([X_att.to(device), zeros.to(device)])
        X_local_source = alpha * X_srouce_feat + (1 - alpha) * X_att
        # return X_local_source,x_out
        return X_local_source

        # ,x_out

    def pad(self, tensor, batch_index):  # 这个函数的作用是对一个批次的序列数据进行填充操作，并生成对应的掩码（mask）
        num_seq = torch.unique(batch_index)
        tensors = [tensor[batch_index == seq_id] for seq_id in num_seq] #这个是将原本叠加在一起的都分开了
        lengths = [len(tensor) for tensor in tensors] #求出最大长度来，对特征向量进行填充，都达到最大节点数20
        lengths = torch.tensor(lengths).to(num_seq.device)
        masks = length_to_mask(lengths) #针对每棵树的20个节点的掩码数组
        return pad_sequence(tensors, batch_first=True), masks.bool() #通过填充让每一颗树的节点数相同;

    def attention_pooling(self, x,batch,batch_size,rootindex): # 单独一个批次的增量注意力池，重点分析对象
        x_source= th.zeros(len(batch), x.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(batch, num_batch))
            x_source[index] = x[rootindex[num_batch]]
        # x_source,_ = self.pad(x_source, batch)
        #x_source = x_batch[0, :]
        # x_i = x_source  # 下面这个是按时间先后顺序的增量注意力的核心
        # x_k = x
        # x_v = x
        # #x_q = x_i.unsqueeze(0).expand(x_k.size(0), -1)
        # x_q = x_i
        # x_c = th.cat([x_q, x_k], dim=1)# 全是原帖子向量的x_q与输入向量x_k进行竖直方向上的拼接。
        # att_i = th.nn.Linear(x_c.size(1),64).to(device)
        # x_c = th.relu(att_i(x_c))  # 激活
        # x_score = self.att_s(x_c)  # 又经过一个线性层，输出维度为1，视为论文里的注意力分数
        # x_score = th.exp(x_score)  # 得到指数计算后的x_score
        #
        # #这里的累加过程看是否可以在多头注意力中进行分开累加，最后合并而不影响效率和精度。
        # x_out = x_v * x_score  # 注意力权重逐个和每个上下文向量相乘，得到加权后的上下文向量矩阵
        # x_out = th.cumsum(x_out, 1).squeeze(1)  # ，进行每个时间段回复帖子与注意力权重相乘后的累加值。可能是论文里的gtk
        # x_score = th.cumsum(x_score, 1) + 1e-10
        # x_out = x_out / x_score  # 得到最终的gtk，经过每个时间段相加后的值

        # x_out= th.cat([x_out, x], dim=1)
        # att_i = th.nn.Linear(x_out.size(1), x.size(1)).to(device)
        # x_out= th.relu(att_i(x_out))
        # x,mask = self.pad

        x_replies = x
        mask = ((x_replies != 0).sum(dim=-1) == 0)
        #x_source_extend,x_out= self.local_attention_network(x_source, x_replies, mask) #利用多头，是否可以输出一个新的根特征集合。
        #x_source_extend= self.local_attention_network(x_source, x_replies, mask)
        x_source_extend= self.local_attention_network(x_source, x_replies, mask)
        x_source_1 = th.zeros(batch_size, x.size(1)).to(device)
        for num_batch in range(batch_size):
            x_source_1[num_batch]= x_source_extend[rootindex[num_batch]]
        x_source_extend=x_source_1
        # ...
        # 处理完成后将其赋值回 x 中对应的位置
        # x1[indices] = x_out
        # # 最后，将 x 中相应位置的多行向量赋给 x_x 的 index 为 true 的位置
        # x_x[index] = th.index_select(x1, 0, indices.to(device))
        # return x_source_extend,x_out
        # return x_source_extend,x_out
        return x_source_extend

    def edge_infer(self, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        # x_i = x[row - 1].unsqueeze(2)
        # x_j = x[col - 1].unsqueeze(1)
        x_i = x[row].unsqueeze(2)
        x_j = x[col].unsqueeze(2)
        x_ij = th.abs(x_i - x_j)
        #x_ij=x_ij.view(1,x_ij.size(1),x_ij.size(0))
        # x_ij=self.stance_network(x_ij)# 通过一个相似度网络来计算节点特征间的相似度，网络內部不太好修改，带结果出来后可以结合立场因素再次调整数值大小来标识节点之间的相似程度。
        sim_val = self.sim_network(x_ij)
        sim_val=sim_val.view(sim_val.size(0),sim_val.size(1))
        # sim_val:(1533,1,64)
        edge_pred = self.fc1(sim_val)  # 通过一个激活函数和全连接层得出一个边的预测值。
        edge_pred = th.sigmoid(edge_pred)
        w_mean = self.W_mean(x_ij)
        w_bias = self.W_bias(x_ij)
        b_mean = self.B_mean(x_ij)
        b_bias = self.B_bias(x_ij)
        sim_val = sim_val.view(sim_val.size(0), sim_val.size(1),1)
        logit_mean = w_mean * sim_val + b_mean  # 相似度*相似度预测值+偏置值  均值
        logit_var = th.log((sim_val ** 2) * th.exp(w_bias) + th.exp(b_bias))  # 方差
        edge_y = th.normal(logit_mean, abs(logit_var))  # 以均值和方差构造正太分布，得到边的标签
        edge_y = th.sigmoid(edge_y)  # 激活层和全连接层
        edge_y= edge_y.view(edge_y.size(0), edge_y.size(1))
        edge_y = self.fc2(edge_y)
        logp_x = F.log_softmax(edge_pred, dim=-1)
        p_y = F.softmax(edge_y, dim=-1)
        edge_loss = self.eval_loss(logp_x, p_y)  # 边的预测值和边的标签之间的损失
        #return edge_loss, th.mean(edge_pred, dim=-1).squeeze(1))
        return edge_loss, th.mean(edge_pred, dim=-1)
        #return edge_loss

    def infer_extend(self, x, logits, batch, batch_size, rootindex):
        x_inder = th.zeros(len(batch), x.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(batch, num_batch))
            x_inder[index] = logits[num_batch]
        return x_inder

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        root=data.root
        root_feature=torch.stack([torch.tensor(item) for item in root]).to(device)
        # x = self.gnn1(x, edge_index)
        # x=self.TSEncoder(x)
        #edge_num_list=data.edge_num
        #推断器模块
        # triple_ids,triple_mask=data.triple_ids,data.triple_mask
        # triple_ids=torch.stack([torch.tensor(item) for item in triple_ids]).to(device)
        # triple_mask = torch.stack([torch.tensor(item) for item in triple_mask]).to(device)
        # logits=self.COMET(triple_ids,triple_mask).view(128,-1)
        # #logits=logits[:,:768]
        # logits=F.relu(logits)
        # logits= F.dropout(logits, training=self.training)
        # logits = self.COMET(data)
        # batch=data.batch
        # batch_size = max(batch) + 1
        # rootindex = data.rootindex
        # # x_infer = self.infer_extend(x, logits, batch, batch_size, rootindex)
        # # x_infer = th.cat((x_infer, x), 1)
        # # x = self.att_inder(x_infer)
        # x1=copy.copy(x.float())
        # x_source_1=self.attention_pooling(x,batch,batch_size,rootindex) #只提取出128条源文本，进行最后的融合
        # x = self.conv1(x, edge_index)
        # edge_loss,edge_pred= self.edge_infer(x, edge_index)  # 在这里计算出边损失
        # x2=copy.copy(x)
        # x_source_2=self.attention_pooling(x, batch, batch_size, rootindex) #同上
        # # x = F.relu(x)
        # # x = F.dropout(x, training=self.training)
        # root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        # #root_extend = th.zeros(len(data.batch), x.size(1)).to(device)
        # for num_batch in range(batch_size):
        #     index = (th.eq(data.batch, num_batch))
        #     root_extend[index] = x1[rootindex[num_batch]]
        # x = th.cat((x,root_extend), 1)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index,edge_weight=edge_pred)
        # #x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # #root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        # root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        # for num_batch in range(batch_size):
        #     index = (th.eq(data.batch, num_batch))
        #     root_extend[index] = x2[rootindex[num_batch]]
        # x = th.cat((x,root_extend), 1)
        # x=scatter_mean(x, batch, dim=0)
        # x_source=th.cat((x_source_1,x_source_2), 1)#只提取出128条源文本来，不进行散列，直接进行源融合
        # #x_source = scatter_mean(x_source, batch, dim=0)
        #
        # # return x,x_source,logits,edge_loss
        # x = th.tanh(x)
        # x_source = th.tanh(x_source)
        # # return x, edge_loss
        # # x_source= th.tanh(x_source)
        # return x,x_source,edge_loss
        # return x
        x = th.tanh(root_feature)
        return x
    #深入层数最多两层，再多也没有意义，原帖子开始的节点一般是舆论的主要影响因素

class BUrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats) #edge_weight是一个可选的Tensor，表示边的权重,可选的参数可考虑使用
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)
        # self.conv1 = TransformerConv(hid_feats, hid_feats // 8, heads=8)
        # self.conv2= TransformerConv(hid_feats*2, out_feats // 8, heads=8)
        # self.conv1 = GATConv(in_feats, 128, heads=6)
        # self.conv2 = GATConv(in_feats * 2, 8, heads=8)
        self.att_i = th.nn.Linear(hid_feats *24, hid_feats)
        self.att_s = th.nn.Linear(hid_feats, 1)
        self.att_inder=th.nn.Linear(hid_feats*2,hid_feats)
        self.mh_attention = MultiheadAttention(input_size=768, output_size=768)
        self.COMET = Source_COMET(in_feats, hid_feats, out_feats)
        #self.attention = Attention(768, 768)
        self.layer_list = OrderedDict()
        self.num_features_list = [768 * r for r in [1]]
        # self.COMET=COMET()
        def creat_network(self, hid_feats=768):  # 添加了一个网络,或许可以从这个网路入手来加入立场因素，进而决定相似度的计算来达到共同学习的目的。
            layer_list = OrderedDict()  # 创建一个字典
            for l in range(len(self.num_features_list)):
                layer_list['conv{}'.format(l)] = th.nn.Conv1d(  # 创建卷积层字典，而且是根据特征列表的长度
                    in_channels=hid_feats,
                    out_channels=hid_feats,
                    kernel_size=1,
                    bias=False)

                layer_list['norm{}'.format(l)] = th.nn.BatchNorm1d(
                    num_features=hid_feats)  # 创建字典归一化函数和激活函数
                layer_list['relu{}'.format(l)] = th.nn.LeakyReLU()

            layer_list['conv_out'] = th.nn.Conv1d(in_channels=hid_feats,
                                                  out_channels=hid_feats,
                                                  kernel_size=1
                                                  )  # 卷积层，最后一层用做输出
            return layer_list

        self.sim_network = th.nn.Sequential(
            creat_network(self))  # 创建了一个名为“sim_val"的神经网络,计算两个节点距离的相似度问题，应该再仔细分析原理，弄明白再处理
        mod_self = self
        mod_self.num_features_list = [hid_feats]

        self.W_mean = th.nn.Sequential(creat_network(mod_self))  # 又创建了一系列的神经网络序列
        self.W_bias = th.nn.Sequential(creat_network(mod_self))
        self.B_mean = th.nn.Sequential(creat_network(mod_self))
        self.B_bias = th.nn.Sequential(creat_network(mod_self))

        self.fc1 = th.nn.Linear(hid_feats, 2, bias=False)
        self.fc2 = th.nn.Linear(hid_feats, 2, bias=False)
        self.dropout = th.nn.Dropout(0.2)
        self.eval_loss = th.nn.KLDivLoss(reduction='batchmean')
        self.bn1 = BatchNorm1d(hid_feats + in_feats)

    def local_attention_network(self, X_source, X_replies, mask):  # mask是一个掩码用于指示哪些元素需要进行注意力计算
        # 可以更加关注源数据和回复数据之间的关联性，从而提高下游任务的性能。
        X_srouce_feat = X_source
        X_replies_feat = X_replies  # 这里可以进一步改成根节点和第一圈的节点的关系，而不必须是全部回复之间的关系，目的得到根节点和离他最近的节点的信息的融合以及他们之间边的路径选择。
        x_out = self.mh_attention(X_replies_feat, X_srouce_feat, X_replies_feat)
        attention = Attention(X_srouce_feat.size(1), X_srouce_feat.size(1))  # 动态注意力机制
        attention = attention.to(device)
        X_att = attention(X_srouce_feat, X_replies_feat, mask=mask)
       # 这里这一部分是源帖子和全部回复之间的注意力分数的信息与原来的根节点帖子信息进行合并，主要得出的是根节点信息。
        X_fuse = th.cat([X_srouce_feat, X_att], dim=-1)
        linear_fuse = th.nn.Linear(X_fuse.size(1), 1)
        linear_fuse = linear_fuse.to(device)
        linear_output = linear_fuse(X_fuse.to(device))
        # alpha = th.sigmoid(linear_fuse(X_fuse.to(device)))
        alpha = th.sigmoid(linear_output)
        # X_att=X_att[:768]
        # zeros = th.zeros(len(X_srouce_feat) -len(X_att) , dtype=X_att.dtype)
        # X_att= th.cat([X_att.to(device), zeros.to(device)])
        X_local_source = alpha * X_srouce_feat + (1 - alpha) * X_att
        # return X_local_source ,x_out
        return X_local_source
        #return x_out

    def attention_pooling(self, x, batch, batch_size, rootindex):  # 单独一个批次的增量注意力池，重点分析对象
        x_source = th.zeros(len(batch), x.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(batch, num_batch))
            x_source[index] = x[rootindex[num_batch]]
        # x_source = x_batch[0, :]
        # x_i = x_source  # 下面这个是按时间先后顺序的增量注意力的核心
        # x_k = x
        # x_v = x
        # #x_q = x_i.unsqueeze(0).expand(x_k.size(0), -1)
        # x_q = x_i
        # x_c = th.cat([x_q, x_k], dim=1)  # 全是原帖子向量的x_q与输入向量x_k进行竖直方向上的拼接。
        # att_i = th.nn.Linear(x_c.size(1), 64).to(device)
        # x_c = th.relu(att_i(x_c))  # 激活
        # #x_c = th.relu(self.att_i(x_c))  # 激活
        # x_score = self.att_s(x_c)  # 又经过一个线性层，输出维度为1，视为论文里的注意力分数
        # x_score = th.exp(x_score)  # 得到指数计算后的x_score
        # x_out = x_v * x_score  # 注意力权重逐个和每个上下文向量相乘，得到加权后的上下文向量矩阵
        # x_out = th.cumsum(x_out, 1).squeeze(1)  # ，进行每个时间段回复帖子与注意力权重相乘后的累加值。可能是论文里的gtk
        # x_score = th.cumsum(x_score, 1) + 1e-10
        # x_out = x_out / x_score  # 得到最终的gtk，经过每个时间段相加后的值

        # x_out = th.cat([x_out, x], dim=1)
        # att_i = th.nn.Linear(x_out.size(1), x.size(1)).to(device)
        # x_out = th.relu(att_i(x_out))

        x_replies = x
        mask = ((x_replies != 0).sum(dim=-1) == 0)
        #x_source_extend,x_out= self.local_attention_network(x_source, x_replies, mask)  # 利用多头，是否可以输出一个新的根特征集合。
        #x_source_extend= self.local_attention_network(x_source, x_replies, mask)
        x_source_extend= self.local_attention_network(x_source, x_replies, mask)
        x_source_1 = th.zeros(batch_size, x.size(1)).to(device)
        for num_batch in range(batch_size):
            x_source_1[num_batch] = x_source_extend[rootindex[num_batch]]
        x_source_extend=x_source_1
        # ...
        # 处理完成后将其赋值回 x 中对应的位置
        # x1[indices] = x_out
        # # 最后，将 x 中相应位置的多行向量赋给 x_x 的 index 为 true 的位置
        # x_x[index] = th.index_select(x1, 0, indices.to(device))
        #return x_source_extend,x_out
        return  x_source_extend
        #return x_out

    def edge_infer(self, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        # x_i = x[row - 1].unsqueeze(2)
        # x_j = x[col - 1].unsqueeze(1)
        x_i = x[row].unsqueeze(2)
        x_j = x[col].unsqueeze(2)
        x_ij = th.abs(x_i - x_j)
        #x_ij=x_ij.view(1,x_ij.size(1),x_ij.size(0))
        # x_ij=self.stance_network(x_ij)# 通过一个相似度网络来计算节点特征间的相似度，网络內部不太好修改，带结果出来后可以结合立场因素再次调整数值大小来标识节点之间的相似程度。
        sim_val = self.sim_network(x_ij)
        sim_val=sim_val.view(sim_val.size(0),sim_val.size(1))
        # sim_val:(1533,1,64)
        edge_pred = self.fc1(sim_val)  # 通过一个激活函数和全连接层得出一个边的预测值。
        edge_pred = th.sigmoid(edge_pred)
        w_mean = self.W_mean(x_ij)
        w_bias = self.W_bias(x_ij)
        b_mean = self.B_mean(x_ij)
        b_bias = self.B_bias(x_ij)
        sim_val = sim_val.view(sim_val.size(0), sim_val.size(1),1)
        logit_mean = w_mean * sim_val + b_mean  # 相似度*相似度预测值+偏置值  均值
        logit_var = th.log((sim_val ** 2) * th.exp(w_bias) + th.exp(b_bias))  # 方差
        edge_y = th.normal(logit_mean, abs(logit_var))  # 以均值和方差构造正太分布，得到边的标签
        edge_y = th.sigmoid(edge_y)  # 激活层和全连接层
        edge_y= edge_y.view(edge_y.size(0), edge_y.size(1))
        edge_y = self.fc2(edge_y)
        logp_x = F.log_softmax(edge_pred, dim=-1)
        p_y = F.softmax(edge_y, dim=-1)
        edge_loss = self.eval_loss(logp_x, p_y)  # 边的预测值和边的标签之间的损失
        #return edge_loss, th.mean(edge_pred, dim=-1).squeeze(1))
        return edge_loss, th.mean(edge_pred, dim=-1)
        #return edge_loss

    def infer_extend(self,x,logits,batch,batch_size,rootindex):
        x_inder = th.zeros(len(batch), x.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(batch, num_batch))
            x_inder[index] = logits[num_batch]
        return x_inder

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        # edge_num_list=data.edge_num
        # 推断器模块
        batch = data.batch
        batch_size = max(batch) + 1
        # triple_ids,triple_mask=data.triple_ids,data.triple_mask
        # triple_ids=torch.stack([torch.tensor(item) for item in triple_ids]).to(device)
        # triple_mask = torch.stack([torch.tensor(item) for item in triple_mask]).to(device)
        # logits=self.COMET(triple_ids,triple_mask).view(batch_size,-1)
        # #logits=logits[:,:768]
        # logits=F.relu(logits)
        # logits= F.dropout(logits, training=self.training)
        # logits= self.COMET(data)
        rootindex = data.rootindex
        # x_infer=self.infer_extend(x, logits, batch,batch_size, rootindex)
        # x_infer=th.cat((x_infer,x),1)
        # x=self.att_inder(x_infer)
        x1 = copy.copy(x.float())
        x_source_1= self.attention_pooling(x, batch, batch_size, rootindex)
        x = self.conv1(x, edge_index)
        edge_loss,edge_pred= self.edge_infer(x, edge_index)  # 在这里计算出边损失
        x2 = copy.copy(x)
        x_source_2= self.attention_pooling(x, batch, batch_size, rootindex)
        # root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index,edge_weight=edge_pred)
        #x = self.conv2(x, edge_index)
        x = F.relu(x)
        # root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)
        x = scatter_mean(x, data.batch, dim=0)
        x_source = th.cat((x_source_1, x_source_2), 1) #这里直接提取出128条源文本来进行直接融合
        #x_source= scatter_mean(x_source, data.batch, dim=0)

        # return x,x_source,logits,edge_loss
        x = th.tanh(x)
        x_source = th.tanh(x_source)
        #return x,edge_loss
        x_source = th.tanh(x_source)
        return x,x_source,edge_loss
        # return x

class Net(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.COMET=Source_COMET(in_feats, hid_feats, out_feats)
        #self.fc=th.nn.Linear((out_feats+hid_feats)*2,4)
        #self.fc = th.nn.Linear(((out_feats + hid_feats)+hid_feats*3)*2, 2)
        self.fc = th.nn.Linear(4736, 3)
        self.fc_comet=th.nn.Linear(768,3)
    def forward(self, data):
        # TD_x,TD_edge_loss = self.TDrumorGCN(data)
        # BU_x,BU_edge_loss = self.BUrumorGCN(data)
        # TD_x,x_source_1,TD_edge_loss= self.TDrumorGCN(data)
        # BU_x,x_source_2,BU_edge_loss= self.BUrumorGCN(data)
        TD_x= self.TDrumorGCN(data)
        # BU_x= self.BUrumorGCN(data)
        # TD_x = th.cat((TD_x, x_source_1), 1)
        # BU_x = th.cat((BU_x, x_source_2), 1)
        # self.x = th.cat((TD_x, BU_x), 1)
        # self.comet_x=self.COMET(data)
        #再和源文本增强矩阵进行一次结合会怎么样？
        #self.x = th.cat((BU_x, TD_x), 1)  # (128,10356)
        # TD_x = th.cat((x_source_1, TD_x), 1)
        # BU_x = th.cat((x_source_2, BU_x), 1)
        # self.x = th.cat((TD_x, BU_x), 1) #带注意力的源融合
        # self.x=self.fc(self.x) #(10356,4)
        # x_source = th.cat((x_source_1, x_source_2), 1)
        # self.comet_x=th.cat((x_source,self.comet_x), 1)
        #self.comet_x=self.fc_comet(self.comet_x)
        self.x = self.fc_comet(TD_x)
        out = F.log_softmax(self.x, dim=1)
        # comet_out=F.log_softmax(self.comet_x,dim=1)
        #return   out,  TD_edge_loss, BU_edge_loss
        # return out,TD_edge_loss,BU_edge_loss
        # return out,TD_edge_loss,BU_edge_loss
        # return out,comet_out,TD_edge_loss,BU_edge_loss
        return out

def train_GCN(x_test,x_train,TDdroprate,BUdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter):
    # if i==1 or i==4 or i==7:
    #     batch_size_test=10
    # batch_size_train=batchsize
    model = Net(768,768,64).to(device)
    BU_params=list(map(id,model.BUrumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    base_params=filter(lambda p:id(p) not in BU_params,model.parameters())
    optimizer = th.optim.Adam([
        {'params':base_params},
        {'params':model.BUrumorGCN.conv1.parameters(),'lr':lr/5},
        {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr/5}
    ], lr=lr, weight_decay=weight_decay)
    model.train()
    criterion = th.nn.NLLLoss()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadBiData(dataname,x_train, x_test,TDdroprate,BUdroprate)  #数据层面
        # train_loader = DataLoader(traindata_list, batch_size=batch_size_train, shuffle=True, num_workers=0,drop_last=True)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=0,drop_last=True)
        # if i==1 or i==4 or i==7:
        #     test_loader = DataLoader(testdata_list, batch_size=batch_size_test, shuffle=False, num_workers=0, drop_last=True)
        # else:
        #     test_loader = DataLoader(testdata_list, batch_size=batch_size_train, shuffle=False, num_workers=0, drop_last=True)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=0,drop_last=True)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            # out_labels,comet_labels,TD_edge_loss,BU_edge_loss= model(Batch_data)
            # out_labels,TD_edge_loss,BU_edge_loss = model(Batch_data)
            out_labels= model(Batch_data)
            #criterion_loss = criterion(out_labels, Batch_data.y)
            #out_labels = out_labels[:128, :] #待修改，不能为了拼接而草率截取
            # finalloss1=F.nll_loss(out_labels,Batch_data.y)
            loss = F.nll_loss(out_labels, Batch_data.y)
            # finalloss2=F.nll_loss(comet_labels,Batch_data.y)
            # finalloss=0.8*finalloss1+0.2*finalloss2 #这里的比例参数待定
            # loss=finalloss1+0.2*TD_edge_loss+0.2*BU_edge_loss
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            # _, pred2 = comet_labels.max(dim=-1)
            # pred= torch.tensor([mode(pair) for pair in zip(pred1, pred2)]).to(device) #使用投票法将两个模型预测结果融合
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            batch_idx = batch_idx + 1
            torch.cuda.empty_cache()
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc)) #平均精度要返回

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
            temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2 ,temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3 = [], [], [], [], [], [], [], [], [],[], [], [], []
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            # val_out,TD_edge_loss,BU_edge_loss= model(Batch_data)
            # val_out,comet_labels,TD_edge_loss,BU_edge_loss= model(Batch_data)
            val_out= model(Batch_data)
            #val_loss1 = F.nll_loss(val_out, Batch_data.y)
            val_loss= F.nll_loss(val_out, Batch_data.y)
            # val_loss2=F.nll_loss(comet_labels,Batch_data.y)
            # val_loss=0.5*val_loss2+0.5*val_loss1
            # val_loss=val_loss1+0.2*TD_edge_loss+0.2*BU_edge_loss
            # val_loss=val_loss1
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=-1)
            # _, val_pred2 = comet_labels.max(dim=-1)
            # val_pred = torch.tensor([mode(pair) for pair in zip(val_pred1, val_pred2)]).to(device)  # 使用投票法将两个模型预测结果融合
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2,Acc3, Prec3, Recll3, F3 = evaluation3class(
                val_pred, Batch_data.y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
                temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3)
            temp_val_accs.append(val_acc)
        torch.cuda.empty_cache()
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))
        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3))]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_Acc_all), np.mean(temp_val_Acc1),
                       np.mean(temp_val_Acc2),np.mean(temp_val_Acc3), np.mean(temp_val_Prec1), np.mean(temp_val_Prec2),np.mean(temp_val_Prec3),
                       np.mean(temp_val_Recll1), np.mean(temp_val_Recll2),np.mean(temp_val_Recll3), np.mean(temp_val_F1), np.mean(temp_val_F2),np.mean(temp_val_F3),
                       model, 'BiGCN', "PHEME")
        accs = np.mean(temp_val_Acc_all)
        acc1 = np.mean(temp_val_Acc1)
        acc2 = np.mean(temp_val_Acc2)
        acc3 = np.mean(temp_val_Acc3)
        pre1 = np.mean(temp_val_Prec1)
        pre2 = np.mean(temp_val_Prec2)
        pre3 = np.mean(temp_val_Prec3)
        rec1 = np.mean(temp_val_Recll1)
        rec2 = np.mean(temp_val_Recll2)
        rec3 = np.mean(temp_val_Recll3)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            acc1 = early_stopping.acc1
            acc2 = early_stopping.acc2
            acc3 = early_stopping.acc3
            pre1 = early_stopping.pre1
            pre2 = early_stopping.pre2
            pre3 = early_stopping.pre3
            rec1 = early_stopping.rec1
            rec2 = early_stopping.rec2
            rec3 = early_stopping.rec3
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            break
    return train_losses, val_losses, train_accs, val_accs, accs, acc1, pre1, rec1, F1, acc2, pre2, rec2, F2,acc3, pre3, rec3, F3


    #     ACC=[]
    #     F1_1=[]
    #     F2_2=[]
    #     F3_3=[]
    #     F4_4=[]
    #     temp_val_losses = []
    #     temp_val_accs = []
    #     temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
    #     temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
    #     temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
    #     temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    #     model.eval()
    #     tqdm_test_loader = tqdm(test_loader)
    #     pred = []  # 存储所有预测结果的列表
    #     true = []
    #     model.eval()  # 切换到评估模式
    #     with torch.no_grad():  # 禁用梯度计算
    #         for Batch_data in tqdm_test_loader:
    #             Batch_data.to(device)
    #             val_out,_,_= model(Batch_data)
    #             val_loss  = F.nll_loss(val_out, Batch_data.y)
    #             temp_val_losses.append(val_loss.item())
    #             _, val_pred = val_out.max(dim=1)
    #             # pred.append(val_pred)
    #             # true.append(Batch_data.y)
    #             correct = val_pred.eq(Batch_data.y).sum().item()
    #             val_acc = correct / len(Batch_data.y)
    #             Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
    #                 val_pred, Batch_data.y)
    #             temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
    #                 Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
    #             temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
    #                 Recll2), temp_val_F2.append(F2), \
    #             temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
    #                 Recll3), temp_val_F3.append(F3), \
    #             temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
    #                 Recll4), temp_val_F4.append(F4)
    #             temp_val_accs.append(val_acc)
    #     # pred = th.cat(pred, dim=0) # 将所有预测结果合并成一个张量
    #     # true = th.cat(true, dim=0) # 将所有真实标签合并成一个张量
    #     # correct = pred.eq(true).sum().item()  # 计算预测正确的样本数量
    #     # acc = correct / len(true)  # 计算精度
    #     val_losses.append(np.mean(temp_val_losses))
    #     val_accs.append(np.mean(temp_val_accs))
    #     print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
    #                                                                        np.mean(temp_val_accs)))
    #     # print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
    #     #                                                                    acc))
    #
    #     res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
    #            'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
    #                                                    np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
    #            'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
    #                                                    np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
    #            'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
    #                                                    np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
    #            'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
    #                                                    np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
    #     ACC.append(np.mean(temp_val_Acc_all))
    #     F1_1.append(np.mean(temp_val_F1))
    #     F2_2.append(np.mean(temp_val_F2))
    #     F3_3.append(np.mean(temp_val_F3))
    #     # res = ['acc:{:.4f}'.format(acc),
    #     #        'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
    #     #                                                np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
    #     #        'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
    #     #                                                np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
    #     #        'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
    #     #                                                np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
    #     #        'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
    #     #                                                np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
    #     print('results:', res)
    #     early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
    #                    np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'BiGCN', 'event')
    #     accs =np.mean(temp_val_accs)
    #     # F1 = np.mean(temp_val_F1)
    #     # F2 = np.mean(temp_val_F2)
    #     # F3 = np.mean(temp_val_F3)
    #     # F4 = np.mean(temp_val_F4)
    #     if early_stopping.early_stop:
    #         print("Early stopping")
    #         accs=early_stopping.accs
    #         # F1=early_stopping.F1
    #         # F2 = early_stopping.F2
    #         # F3 = early_stopping.F3
    #         F4 = early_stopping.F4
    #         break
    #
    # return train_losses , val_losses ,train_accs, val_accs,max(ACC),max(F1_1),max(F2_2),max(F3_3),F4

if __name__ == '__main__':
 # rumour_thread_size=638+1067+697
 # rumour_false=638
 # rumour_true=1067
 # rumour_unverified=697
 # rumour_false_weight=rumour_false/rumour_thread_size
 # rumour_true_weight=rumour_true/rumour_thread_size
 # rumour_unverified_weight=rumour_unverified/rumour_thread_size
 lr=0.0001
weight_decay=1e-4
patience=10
n_epochs=200
batchsize=128
TDdroprate=0.2
BUdroprate=0.2
#datasetname="charliehebdo" #"Twitter15"、"Twitter16"
# datasetname= ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting', 'sydneysiege']
datasetname="Pheme"
#five_folders=['ebola-essien','gurlitt','prince-toronto','putinmissing']
iterations=1
model="GCN"
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
test_accs = []
# NR_F1 = []
# FR_F1 = []
# TR_F1 = []
# UR_F1 = []
ACC1, ACC2,ACC3, PRE1, PRE2,PRE3, REC1, REC2,REC3, F1, F2 ,F3= [], [], [], [], [], [], [], [],[], [], [], []
for iter in range(iterations):
    fold0_x_test, fold0_x_train, \
    fold1_x_test, fold1_x_train, \
    fold2_x_test, fold2_x_train, \
    fold3_x_test, fold3_x_train, \
    fold4_x_test, fold4_x_train = load5foldData()
    #treeDic=loadTree(datasetname)
    train_losses, val_losses, train_accs, val_accs, accs_0, acc1_0, pre1_0, rec1_0,  F1_0, acc2_0, pre2_0, rec2_0, F2_0,acc3_0, pre3_0, rec3_0, F3_0 = train_GCN(
                                                                                                fold0_x_test,
                                                                                                fold0_x_train,
                                                                                                TDdroprate, BUdroprate, lr,
                                                                                                weight_decay,
                                                                                                patience,
                                                                                                n_epochs,
                                                                                                batchsize,
                                                                                                datasetname,
                                                                                                iter)  # 0 1 2 3
    torch.cuda.empty_cache()

    train_losses, val_losses, train_accs, val_accs, accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1,acc3_1, pre3_1, rec3_1, F3_1= train_GCN(
                                                                                               fold1_x_test,
                                                                                               fold1_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter) #0 1 2 4
    torch.cuda.empty_cache()
    train_losses, val_losses, train_accs, val_accs, accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2, acc3_2, pre3_2, rec3_2, F3_2 = train_GCN(
                                                                                               fold2_x_test,
                                                                                               fold2_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)# 0 1 3 4
    torch.cuda.empty_cache()
    train_losses, val_losses, train_accs, val_accs, accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3,acc3_3, pre3_3, rec3_3, F3_3 = train_GCN(
                                                                                               fold3_x_test,
                                                                                               fold3_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)# 0 2 3 4
    torch.cuda.empty_cache()
    train_losses, val_losses, train_accs, val_accs, accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4,acc3_4, pre3_4, rec3_4, F3_4 = train_GCN(
                                                                                               fold4_x_test,
                                                                                               fold4_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)# 1 2 3 4
    torch.cuda.empty_cache()

    # test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4) / 5)
    # ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
    # ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
    # PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
    # PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
    # REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
    # REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
    # F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
    # F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4) / 5)
    ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
    ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
    ACC3.append((acc3_0 + acc3_1 + acc3_2 + acc3_3 + acc3_4) / 5)
    PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
    PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
    PRE3.append((pre3_0 + pre3_1 + pre3_2 + pre3_3 + pre3_4) / 5)
    REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
    REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
    REC3.append((rec3_0 + rec3_1 + rec3_2 + rec3_3 + rec3_4) / 5)
    F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
    F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    F3.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
print("Total_Test_Accuracy: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
    sum(test_accs) / iterations, sum(F1) /iterations, sum(F2) /iterations, sum(F3) / iterations))
# print("Twitter:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|acc3: {:.4f}|pre1: {:.4f}|pre2: {:.4f}|pre3: {:.4f}"
#       "|rec1: {:.4f}|rec2: {:.4f}|rec3: {:.4f}|F1: {:.4f}|F2: {:.4f}|F3: {:.4f}|mF:{:.4f}|wF:{:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations,
#                                                                 sum(ACC2) / iterations,sum(ACC3) / iterations, sum(PRE1) / iterations,
#                                                                 sum(PRE2) / iterations,sum(PRE3) / iterations,
#                                                                 sum(REC1) / iterations, sum(REC2) / iterations,sum(REC3) / iterations,
#                                                                 sum(F1) / iterations, sum(F2) / iterations,sum(F3) / iterations,((sum(F1) / iterations+sum(F2) / iterations+sum(F3) / iterations)/3),(sum(F1) / iterations)*rumour_true_weight+(sum(F2) / iterations)*rumour_false_weight+(sum(F3) / iterations)*rumour_unverified_weight))

# if __name__ == '__main__':
#  rumour_thread_size=638+1067+697
#  rumour_false=638
#  rumour_true=1067
#  rumour_unverified=697
#  rumour_false_weight=rumour_false/rumour_thread_size
#  rumour_true_weight=rumour_true/rumour_thread_size
#  rumour_unverified_weight=rumour_unverified/rumour_thread_size
#  lr=0.0002
# weight_decay=1e-4
# patience=10
# n_epochs=200
# batchsize=128
# TDdroprate=0.2
# BUdroprate=0.2
# #datasetname="charliehebdo" #"Twitter15"、"Twitter16"
# # datasetname= ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting', 'sydneysiege']
# datasetname=['charliehebdo','ebola-essien','ferguson', 'germanwings-crash','gurlitt', 'ottawashooting','prince-toronto','putinmissing','sydneysiege']
# #five_folders=['ebola-essien','gurlitt','prince-toronto','putinmissing']
# iterations=1
# model="GCN"
# device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
# test_accs = []
# # NR_F1 = []
# # FR_F1 = []
# # TR_F1 = []
# # UR_F1 = []
# ACC1, ACC2,ACC3, PRE1, PRE2,PRE3, REC1, REC2,REC3, F1, F2 ,F3= [], [], [], [], [], [], [], [],[], [], [], []
# for iter in range(iterations):
#     fold0_x_test, fold0_x_train= load5foldData(datasetname,i=0)
#     fold1_x_test,  fold1_x_train= load5foldData(datasetname,i=1)
#     fold2_x_test, fold2_x_train= load5foldData(datasetname,i=2)
#     fold3_x_test, fold3_x_train= load5foldData(datasetname,i=3)
#     fold4_x_test,fold4_x_train = load5foldData(datasetname,i=4)
#     fold5_x_test, fold5_x_train = load5foldData(datasetname, i=5)
#     fold6_x_test, fold6_x_train = load5foldData(datasetname, i=6)
#     fold7_x_test, fold7_x_train = load5foldData(datasetname, i=7)
#     fold8_x_test, fold8_x_train = load5foldData(datasetname, i=8)
#     #treeDic=loadTree(datasetname)
#     train_losses, val_losses, train_accs, val_accs, accs_0, acc1_0, pre1_0, rec1_0,  F1_0, acc2_0, pre2_0, rec2_0, F2_0,acc3_0, pre3_0, rec3_0, F3_0 = train_GCN(
#                                                                                                 fold0_x_test,
#                                                                                                 fold0_x_train,
#                                                                                                 TDdroprate, BUdroprate, lr,
#                                                                                                 weight_decay,
#                                                                                                 patience,
#                                                                                                 n_epochs,
#                                                                                                 batchsize,
#                                                                                                 datasetname,
#                                                                                                 iter, i=0)  # 0 1 2 3
#     torch.cuda.empty_cache()
#
#     train_losses, val_losses, train_accs, val_accs, accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1,acc3_1, pre3_1, rec3_1, F3_1= train_GCN(
#                                                                                                fold1_x_test,
#                                                                                                fold1_x_train,
#                                                                                                TDdroprate,BUdroprate, lr,
#                                                                                                weight_decay,
#                                                                                                patience,
#                                                                                                n_epochs,
#                                                                                                batchsize,
#                                                                                                datasetname,
#                                                                                                iter,i=1) #0 1 2 4
#     torch.cuda.empty_cache()
#     train_losses, val_losses, train_accs, val_accs, accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2, acc3_2, pre3_2, rec3_2, F3_2 = train_GCN(
#                                                                                                fold2_x_test,
#                                                                                                fold2_x_train,
#                                                                                                TDdroprate,BUdroprate, lr,
#                                                                                                weight_decay,
#                                                                                                patience,
#                                                                                                n_epochs,
#                                                                                                batchsize,
#                                                                                                datasetname,
#                                                                                                iter,i=2)# 0 1 3 4
#     torch.cuda.empty_cache()
#     train_losses, val_losses, train_accs, val_accs, accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3,acc3_3, pre3_3, rec3_3, F3_3 = train_GCN(
#                                                                                                fold3_x_test,
#                                                                                                fold3_x_train,
#                                                                                                TDdroprate,BUdroprate, lr,
#                                                                                                weight_decay,
#                                                                                                patience,
#                                                                                                n_epochs,
#                                                                                                batchsize,
#                                                                                                datasetname,
#                                                                                                iter,i=3)# 0 2 3 4
#     torch.cuda.empty_cache()
#     train_losses, val_losses, train_accs, val_accs, accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4,acc3_4, pre3_4, rec3_4, F3_4 = train_GCN(
#                                                                                                fold4_x_test,
#                                                                                                fold4_x_train,
#                                                                                                TDdroprate,BUdroprate, lr,
#                                                                                                weight_decay,
#                                                                                                patience,
#                                                                                                n_epochs,
#                                                                                                batchsize,
#                                                                                                datasetname,
#                                                                                                iter,i=4)# 1 2 3 4
#     torch.cuda.empty_cache()
#     train_losses, val_losses, train_accs, val_accs, accs_5, acc1_5, pre1_5, rec1_5, F1_5, acc2_5, pre2_5, rec2_5, F2_5,acc3_5, pre3_5, rec3_5, F3_5 = train_GCN(
#                                                                                                                 fold5_x_test,
#                                                                                                                 fold5_x_train,
#                                                                                                                 TDdroprate, BUdroprate, lr,
#                                                                                                                 weight_decay,
#                                                                                                                 patience,
#                                                                                                                 n_epochs,
#                                                                                                                 batchsize,
#                                                                                                                 datasetname,
#                                                                                                                 iter, i=5)  # 1 2 3 4
#
#     train_losses, val_losses, train_accs, val_accs, accs_6, acc1_6, pre1_6, rec1_6, F1_6, acc2_6, pre2_6, rec2_6, F2_6,acc3_6, pre3_6, rec3_6, F3_6 = train_GCN(
#                                                                                                                             fold6_x_test,
#                                                                                                                             fold6_x_train,
#                                                                                                                             TDdroprate, BUdroprate, lr,
#                                                                                                                             weight_decay,
#                                                                                                                             patience,
#                                                                                                                             n_epochs,
#                                                                                                                             batchsize,
#                                                                                                                             datasetname,
#                                                                                                                             iter, i=6)  # 1 2 3 4
#
#     train_losses, val_losses, train_accs, val_accs, accs_7, acc1_7, pre1_7, rec1_7, F1_7, acc2_7, pre2_7, rec2_7, F2_7,acc3_7, pre3_7, rec3_7, F3_7 = train_GCN(
#                                                                                                                             fold7_x_test,
#                                                                                                                             fold7_x_train,
#                                                                                                                             TDdroprate, BUdroprate, lr,
#                                                                                                                             weight_decay,
#                                                                                                                             patience,
#                                                                                                                             n_epochs,
#                                                                                                                             batchsize,
#                                                                                                                             datasetname,
#                                                                                                                             iter, i=7)  # 1 2 3 4
#     train_losses, val_losses, train_accs, val_accs, accs_8, acc1_8, pre1_8, rec1_8, F1_8, acc2_8, pre2_8, rec2_8, F2_8,acc3_8, pre3_8, rec3_8, F3_8  = train_GCN(
#                                                                                                                         fold8_x_test,
#                                                                                                                         fold8_x_train,
#                                                                                                                         TDdroprate, BUdroprate, lr,
#                                                                                                                         weight_decay,
#                                                                                                                         patience,
#                                                                                                                         n_epochs,
#                                                                                                                         batchsize,
#                                                                                                                         datasetname,
#                                                                                                                         iter, i=8)  # 1 2 3 4
#     # test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4) / 5)
#     # ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
#     # ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
#     # PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
#     # PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
#     # REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
#     # REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
#     # F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
#     # F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
#     test_accs.append((accs_0 + accs_1 + accs_2 + accs_3 + accs_4+accs_5+accs_6+accs_7+accs_8) / 9)
#     ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4+acc1_5+acc1_6+acc1_7+acc1_8) / 9)
#     ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4+acc2_5+acc2_6+acc2_7+acc2_8) / 9)
#     ACC3.append((acc3_0 + acc3_1 + acc3_2 + acc3_3 + acc3_4 + acc3_5 + acc3_6 + acc3_7 + acc3_8) / 9)
#     PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4+pre1_5+pre1_6+pre1_7+pre1_8) / 9)
#     PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4+pre2_5+pre2_6+pre2_7+pre2_8) / 9)
#     PRE3.append((pre3_0 + pre3_1 + pre3_2 + pre3_3 + pre3_4 + pre3_5 + pre3_6 + pre3_7 + pre3_8) / 9)
#     REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4+rec1_5+rec1_6+rec1_7+rec1_8) / 9)
#     REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4+rec2_5+rec2_6+rec2_7+rec2_8) / 9)
#     REC3.append((rec3_0 + rec3_1 + rec3_2 + rec3_3 + rec3_4 + rec3_5 + rec3_6 + rec3_7 + rec3_8) / 9)
#     F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4+F1_5+F1_6+F1_7+F1_8) / 9)
#     F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4+F2_5+F2_6+F2_7+F2_8) / 9)
#     F3.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4 + F3_5 + F3_6 + F3_7 + F3_8) / 9)
# print("Twitter:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|acc3: {:.4f}|pre1: {:.4f}|pre2: {:.4f}|pre3: {:.4f}"
#       "|rec1: {:.4f}|rec2: {:.4f}|rec3: {:.4f}|F1: {:.4f}|F2: {:.4f}|F3: {:.4f}|mF:{:.4f}|wF:{:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations,
#                                                                 sum(ACC2) / iterations,sum(ACC3) / iterations, sum(PRE1) / iterations,
#                                                                 sum(PRE2) / iterations,sum(PRE3) / iterations,
#                                                                 sum(REC1) / iterations, sum(REC2) / iterations,sum(REC3) / iterations,
#                                                                 sum(F1) / iterations, sum(F2) / iterations,sum(F3) / iterations,((sum(F1) / iterations+sum(F2) / iterations+sum(F3) / iterations)/3),(sum(F1) / iterations)*rumour_true_weight+(sum(F2) / iterations)*rumour_false_weight+(sum(F3) / iterations)*rumour_unverified_weight))

# train_losses, val_losses, train_accs, val_accs0, accs5, F1_5, F2_5, F3_5, F4_5 = train_GCN(
    #                                                                                         fold5_x_test,
    #                                                                                         fold5_x_train,
    #                                                                                         TDdroprate, BUdroprate,
    #                                                                                         lr, weight_decay,
    #                                                                                         patience,
    #                                                                                         n_epochs,
    #                                                                                         batchsize,
    #                                                                                         datasetname,
    #                                                                                         iter, i=5)  # 0 1 2 3

    # train_losses, val_losses, train_accs, val_accs0, accs6, F1_6, F2_6, F3_6, F4_6 = train_GCN(
    #                                                                                             fold6_x_test,
    #                                                                                             fold6_x_train,
    #                                                                                             TDdroprate, BUdroprate,
    #                                                                                             lr, weight_decay,
    #                                                                                             patience,
    #                                                                                             n_epochs,
    #                                                                                             batchsize,
    #                                                                                             datasetname,
    #                                                                                             iter, i=6)  # 0 1 2 3
    # #
    # train_losses, val_losses, train_accs, val_accs0, accs7, F1_7, F2_7, F3_7, F4_7 = train_GCN(
    #                                                                                         fold7_x_test,
    #                                                                                         fold7_x_train,
    #                                                                                         TDdroprate, BUdroprate,
    #                                                                                         lr, weight_decay,
    #                                                                                         patience,
    #                                                                                         n_epochs,
    #                                                                                         batchsize,
    #                                                                                         datasetname,
    #                                                                                         iter, i=7)  # 0 1 2 3

    # train_losses, val_losses, train_accs, val_accs0, accs8, F1_8, F2_8, F3_8, F4_8 = train_GCN(
    #                                                                                         fold8_x_test,
    #                                                                                         fold8_x_train,
    #                                                                                         TDdroprate, BUdroprate,
    #                                                                                         lr, weight_decay,
    #                                                                                         patience,
    #                                                                                         n_epochs,
    #                                                                                         batchsize,
    #                                                                                         datasetname,
    #                                                                                         iter, i=8)  # 0 1 2 3
#     test_accs.append((accs0 + accs1 + accs2 + accs3 + accs4) / 5)
#     NR_F1.append((F1_0 + F1_1 + F1_2 + F1_3 + F1_4) / 5)
#     FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
#     TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
#     UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
# print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
#         sum(test_accs) / iterations, sum(NR_F1) / iterations, sum(FR_F1) / iterations, sum(TR_F1) / iterations,
#         sum(UR_F1) / iterations))
#     # print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
#     #     test_accs[iter], sum(NR_F1) / iterations, sum(FR_F1) / iterations, sum(TR_F1) / iterations,
#     #     sum(UR_F1) / iterations))
#     # print("\n")
# print("Total_F1:{:.4f}".format((NR_F1[iter]+FR_F1[iter])/2))

#     test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)
#     NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
#     FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
#     TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
#     UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
# print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(sum(test_accs) / iterations, sum(NR_F1) /iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations))
#


