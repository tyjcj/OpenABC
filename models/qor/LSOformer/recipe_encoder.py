# models/recipe_encoder.py - Recipe序列编码器模块
import torch
import torch.nn as nn
import math


class RecipeEncoder(nn.Module):
    def __init__(self, num_heuristics, embedding_dim):
        """
        Recipe编码器 - 将启发式算法索引转换为嵌入向量
        Recipe是优化命令的序列，例如：
        "balance;rewrite;refactor" -> [0, 1, 2] -> embeddings
        这里使用简单的Embedding层，每个启发式算法对应一个可学习的向量

        Args:
            num_heuristics: 启发式算法的总数（词汇表大小）
            embedding_dim: 嵌入向量的维度（通常为2*hidden_dim）
        """
        super(RecipeEncoder, self).__init__()
        # 可学习的嵌入矩阵：[num_heuristics, embedding_dim]
        self.embedding = nn.Embedding(num_heuristics, embedding_dim)

    def forward(self, recipe_indices):
        """
        将Recipe索引转换为嵌入向量

        Args: recipe_indices: 启发式算法索引 [batch_size, seq_len]
              例如：[[0, 1, 2], [1, 2, 0]] 表示两个Recipe序列

        Returns: Recipe嵌入 [batch_size, seq_len, embedding_dim]
        """
        return self.embedding(recipe_indices)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Transformer位置编码 - 标准的正弦余弦位置编码

        由于Recipe序列的顺序很重要（不同的优化顺序会产生不同的结果），
        需要位置编码来让模型理解序列中的位置信息

        使用sin/cos函数的原因：
        1. 能够处理任意长度的序列
        2. 不同位置的编码向量彼此不同
        3. 相近位置的编码向量相似度较高

        Args:
            d_model: 模型维度（必须与embedding_dim一致）
            dropout: Dropout比率
            max_len: 支持的最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]

        # 计算分母项：10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数位置使用sin，奇数位置使用cos
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(pos/10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(pos/10000^(2i/d_model))

        pe = pe.unsqueeze(0)  # 添加batch维度：[1, max_len, d_model]

        # 注册为buffer，不参与梯度更新但会随模型一起保存/加载
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        为输入序列添加位置编码

        Args: x: 输入张量 [batch_size, seq_len, d_model]
                 通常是Recipe的嵌入向量

        Returns: 添加位置编码后的输出 [batch_size, seq_len, d_model]
        """
        # 将位置编码加到输入上
        # pe[:, :x.size(1), :] 确保位置编码的长度与输入序列长度匹配
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)  # 用dropout防止过拟合