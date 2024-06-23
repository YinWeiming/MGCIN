
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class MultiheadAttention(nn.Module):

    def __init__(self, input_size, output_size, d_k=16, d_v=16, num_heads=8, is_layer_norm=False, attn_dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size
        self.att_s = torch.nn.Linear(64, 1)
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, num_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, num_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, num_heads * d_v))
        self.W_o = nn.Parameter(torch.Tensor(d_v*num_heads, input_size))

        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.linear3 = nn.Linear(input_size, output_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.W_q)
        init.xavier_uniform_(self.W_k)
        init.xavier_uniform_(self.W_v)
        init.xavier_uniform_(self.W_o)
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        init.xavier_uniform_(self.linear3.weight)

    def feed_forword_layer(self, X):
        lay1 = F.relu(self.linear1(X))
        lay1 = self.dropout(lay1)

        output = self.linear2(lay1)
        return output

    def scaled_dot_product_attention(self, Q, K, V, key_padding_mask, episilon=1e-6):
        '''
        :param Q: (*, max_q_words, num_heads, input_size)
        :param K: (*, max_k_words, num_heads, input_size)
        :param V: (*, max_v_words, num_heads, input_size)
        :param episilon:
        :return:
        '''
        x_i = K  # 下面这个是按时间先后顺序的增量注意力的核心
        x_k = Q
        x_v = V
        #x_q = x_i.unsqueeze(0).expand(x_k.size(0), -1)
        x_q = x_i
        x_c = torch.cat([x_q, x_k], dim=2)  # 全是原帖子向量的x_q与输入向量x_k进行竖直方向上的拼接。
        att_i = torch.nn.Linear(x_c.size(2), 64).to(device)
        x_c = torch.relu(att_i(x_c))  # 激活
        #x_c = th.relu(self.att_i(x_c))  # 激活
        x_score = self.att_s(x_c)  # 又经过一个线性层，输出维度为1，视为论文里的注意力分数
        x_score = torch.exp(x_score)  # 得到指数计算后的x_score
        x_out = x_v * x_score  # 注意力权重逐个和每个上下文向量相乘，得到加权后的上下文向量矩阵
        x_out = torch.cumsum(x_out, 1).squeeze(1)  # ，进行每个时间段回复帖子与注意力权重相乘后的累加值。可能是论文里的gtk
        x_score = torch.cumsum(x_score, 1) + 1e-10
        x_out = x_out / x_score  # 得到最终的gtk，经过每个时间段相加后的值

        # x_out = th.cat([x_out, x], dim=1)
        # att_i = th.nn.Linear(x_out.size(1), x.size(1)).to(device)
        # x_out = th.relu(att_i(x_out))

        #
        # temperature = self.d_k ** 0.5
        # Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        #
        # if key_padding_mask is not None:
        #     bsz, src_len = Q.size(0) // self.num_heads, Q.size(1)
        #     tgt_len = V.size(1)
        #     Q_K = Q_K.view(bsz, self.num_heads, tgt_len, src_len)
        #     key_padding_mask = key_padding_mask.unsqueeze(dim=1).unsqueeze(dim=2)
        #     Q_K = Q_K.masked_fill(key_padding_mask, -2 ** 32 + 1)
        #     Q_K = Q_K.view(bsz * self.num_heads, tgt_len, src_len)
        #
        # Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_q_words, max_k_words)
        # Q_K_score = self.dropout(Q_K_score)
        #
        # V_att = Q_K_score.bmm(V)  # (*, max_q_words, input_size)
        # return V_att
        return x_out


    def multi_head_attention(self, Q, K, V, key_padding_mask):
       
        # bsz, q_len= Q.size()
        # bsz, k_len= K.size()
        # bsz, v_len= V.size()
        q_len,dim_Q= Q.size()
        k_len,dim_k= K.size()
        v_len,dim_v= V.size()
        # Q_ = Q.matmul(self.W_q).view(q_len, self.num_heads, self.d_k)
        # Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.num_heads, self.d_k) #这里d_k,d_v表示输入尺寸，Q，K,V分别与权重矩阵相乘。
        # K_ = K.matmul(self.W_k).view(bsz, k_len, self.num_heads, self.d_k)
        # V_ = V.matmul(self.W_v).view(bsz, v_len, self.num_heads, self.d_v)
        Q_ = Q.matmul(self.W_q).view(q_len, self.num_heads, self.d_k)  # 这里d_k,d_v表示输入尺寸，Q，K,V分别与权重矩阵相乘。
        K_ = K.matmul(self.W_k).view(k_len, self.num_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(v_len, self.num_heads, self.d_v)
        # Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.num_heads, q_len, self.d_k)
        # K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.num_heads, q_len, self.d_k)
        # V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.num_heads, q_len, self.d_v)
        # Q_ = Q_.permute(0, 2, 1).contiguous().view(self.num_heads, q_len, self.d_k)
        # K_ = K_.permute(0, 2, 1).contiguous().view(self.num_heads, q_len, self.d_k)
        # V_ = V_.permute(0, 2, 1).contiguous().view(self.num_heads, q_len, self.d_v)
        V_att = self.scaled_dot_product_attention(Q_, K_, V_, key_padding_mask)
        #V_att = V_att.view(bsz, self.num_heads, q_len, self.d_v)
        V_att = V_att.view(self.num_heads, q_len, self.d_v)
        #V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.num_heads*self.d_v)
        V_att = V_att.permute(0, 2, 1).contiguous().view(q_len, self.num_heads * self.d_v)
        output = self.dropout(V_att.matmul(self.W_o)) # (batch_size, max_q_words, input_size)
        return output


    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        '''
        :param query: (batch_size, max_q_words, input_size)
        :param key: (batch_size, max_k_words, input_size)
        :param value: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        #bsz, src_len = query.size()
        bsz, src_len= query.size()
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        V_att = self.multi_head_attention(query, key, value, key_padding_mask)

        # if self.is_layer_norm:
        #     X = self.layer_morm(query + V_att)  # (batch_size, max_r_words, embedding_dim)
        #     output = self.layer_morm(self.feed_forword_layer(X) + X)
        # else:
        #     X = query + V_att
        #     output = self.feed_forword_layer(X) + X
        #
        # output = self.linear3(output)
        # return output
        return V_att
