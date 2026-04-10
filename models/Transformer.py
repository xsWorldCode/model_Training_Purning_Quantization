import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)# 定义dropout层，设置丢弃概率,对10%的神经元进行随机失活
        self.softmax = nn.Softmax(dim=-1) # 将得分转换成概率
    def forward(self, Q, K, V, mask=None):
        # X:batch_size, seq_len, d_model
        # batch:一次处理的样本数量，一个句子中token数量；seq_len:序列长度；d_model:每个元素的特征维度
        # Q,query向量 维度：batch,head,seq_len_q,d_k
        # K,key向量 维度：batch,head,seq_len_k,d_k
        # V,value向量 维度：batch,head,seq_len_v,d_v
        d_k = Q.size(-1) # 获取查询向量的维度
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(torch.tensor(d_k, dtype=torch.float32))# 计算注意力得分，使用缩放点积注意力机制，除以sqrt(d_k)进行缩放
        # scores维度：batch,head,seq_len_q,seq_len_k
        # 如果提供了mask，通过mask==0找到需要屏蔽的位置,mask_fill则将mask位置的得分设置为负无穷，确保这些位置在softmax后得到的概率为0
        #mask==0 表示被屏蔽，mask==1 表示当前可见
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        att = self.softmax(scores) # 对得分进行softmax，得到注意力权重
        #
        att = self.dropout(att) # 对注意力权重进行dropout，增加模型的鲁棒性
