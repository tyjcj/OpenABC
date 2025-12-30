# models/graph_encoder.py - AIG图编码器模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, BatchNorm


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, task_type="delay"):
        """
        AIG图的图编码器 - 根据任务类型选择不同架构

        Args: input_dim: 输入节点特征维度（通常是节点类型+反相输入数量）
              hidden_dim: 隐藏层维度
              task_type: 任务类型，"delay"使用GCN，"area"使用GIN
        """
        super(GraphEncoder, self).__init__()
        self.task_type = task_type.lower()

        if self.task_type == "delay":
            # Delay预测：2层GCN，hidden_dim=32
            self.gcn1 = GCNConv(input_dim, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        elif self.task_type == "area":
            # Area预测：10层GIN with batch normalization
            self.gin_layers = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            # 第一层
            mlp1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gin_layers.append(GINConv(mlp1))
            self.batch_norms.append(BatchNorm(hidden_dim))

            # 中间8层
            for _ in range(8):
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.gin_layers.append(GINConv(mlp))
                self.batch_norms.append(BatchNorm(hidden_dim))

            # 最后一层
            mlp_final = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gin_layers.append(GINConv(mlp_final))

    def forward(self, x, edge_index):
        """
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
        Returns: 节点嵌入 [num_nodes, hidden_dim]
        """
        if self.task_type == "delay":
            x = F.relu(self.gcn1(x, edge_index))
            x = self.gcn2(x, edge_index)
            return x
        elif self.task_type == "area":
            for i, (gin_layer, batch_norm) in enumerate(zip(self.gin_layers[:-1], self.batch_norms)):
                x = gin_layer(x, edge_index)
                x = batch_norm(x)
                x = F.relu(x)
            x = self.gin_layers[-1](x, edge_index)
            return x


class LevelWisePooling(nn.Module):
    def __init__(self, hidden_dim):
        """
        层级池化模块 - 论文的核心创新之一

        DAG结构的AIG图具有天然的层级结构：
        - Level 0: 主输入（Primary Inputs, PI）
        - Level 1: 连接到PI的门
        - Level 2: 连接到Level 1的门
        - ...
        - Level max: 主输出（Primary Outputs, PO）
        传统图池化会丢失这种层级结构信息，level-wise pooling保留了这种偏差

        Args: hidden_dim: 节点嵌入的隐藏维度
        """
        super(LevelWisePooling, self).__init__()
        self.hidden_dim = hidden_dim

    # def forward(self, node_embeddings, node_depths):
    #     """
    #     执行层级池化
    #
    #     对于每一层的节点，同时应用mean pooling和max pooling：
    #     - Mean pooling: 捕获该层节点的平均特征
    #     - Max pooling: 捕获该层的显著特征
    #     - 拼接两者：得到更丰富的层级表示
    #
    #     Args:
    #         node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
    #         node_depths: 每个节点的深度 [num_nodes] 或 [batch, num_nodes]
    #
    #     Returns:
    #         层级嵌入序列 [max_depth, 2*hidden_dim]
    #         - 每一行代表一个层级的嵌入
    #         - 维度翻倍是因为拼接了mean和max pooling
    #     """
    #     # 确保node_depths的维度正确
    #     if node_depths.dim() > 1:
    #         node_depths = node_depths.squeeze(0)
    #
    #     # 获取最大深度，确定需要多少个层级
    #     max_depth = int(node_depths.max().item()) + 1
    #     level_embeddings = []
    #
    #     # 对每一层分别进行池化
    #     for level in range(max_depth):
    #         # 获取当前层级的节点掩码
    #         level_mask = (node_depths == level)
    #         if level_mask.sum() > 0:
    #             # 提取当前层级的节点嵌入
    #             level_nodes = node_embeddings[level_mask]
    #
    #             # 应用mean和max池化
    #             mean_pool = torch.mean(level_nodes, dim=0)  # 平均池化 [hidden_dim]
    #             max_pool = torch.max(level_nodes, dim=0)[0]  # 最大池化 [hidden_dim]
    #
    #             # 拼接两种池化结果
    #             level_emb = torch.cat([mean_pool, max_pool], dim=0)  # [2*hidden_dim]
    #             level_embeddings.append(level_emb)
    #         else:
    #             # 如果某层没有节点（理论上不应该发生），添加零嵌入
    #             level_embeddings.append(torch.zeros(2 * self.hidden_dim,
    #                                                 device=node_embeddings.device))
    #
    #     # 堆叠所有层级嵌入，形成序列
    #     # 输出形状：[max_depth, 2*hidden_dim]
    #     # 这个序列将作为Transformer解码器的memory（key, value）
    #     return torch.stack(level_embeddings)
    def forward(self, node_embeddings, node_depths):
        """
        执行层级池化

        对于每一层的节点，同时应用mean pooling和max pooling：
        - Mean pooling: 捕获该层节点的平均特征
        - Max pooling: 捕获该层的显著特征
        - 拼接两者：得到更丰富的层级表示

        Args:
            node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
            node_depths: 每个节点的深度 [num_nodes] 或 [batch, num_nodes]

        Returns:
            层级嵌入序列 [max_depth, 2*hidden_dim]
            - 每一行代表一个层级的嵌入
            - 维度翻倍是因为拼接了mean和max pooling
        """
        # 确保node_depths的维度正确
        if node_depths.dim() > 1:
            node_depths = node_depths.squeeze(0)

        # 获取最大深度，确定需要多少个层级
        max_depth = int(node_depths.max().item()) + 1
        level_embeddings = []

        # 对每一层分别进行池化
        for level in range(max_depth):
            # 获取当前层级的节点掩码
            level_mask = (node_depths == level)
            if level_mask.sum() > 0:
                # 提取当前层级的节点嵌入
                level_nodes = node_embeddings[level_mask]

                # 应用mean和max池化
                mean_pool = torch.mean(level_nodes, dim=0)  # 平均池化 [hidden_dim]
                max_pool = torch.max(level_nodes, dim=0)[0]  # 最大池化 [hidden_dim]

                # 拼接两种池化结果
                level_emb = torch.cat([mean_pool, max_pool], dim=0)  # [2*hidden_dim]
                level_embeddings.append(level_emb)
            else:
                # 如果某层没有节点（理论上不应该发生），添加零嵌入
                level_embeddings.append(torch.zeros(2 * self.hidden_dim,
                                                    device=node_embeddings.device))

        # 堆叠所有层级嵌入，形成序列
        # 输出形状：[max_depth, 2*hidden_dim]
        # 这个序列将作为Transformer解码器的memory（key, value）
        return torch.stack(level_embeddings)