import logging
import sys
import torch
import torch.nn as nn
import subgraph_sampling as sampler
import torch.nn.functional as F
import utils
from tqdm import tqdm
import torch_geometric as tg
import torch.optim as optim
import feature_extractor as extractor
import data_loader as loader
from torch_geometric.data import DataLoader
import clip
    
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from evaluate import *


# Define the C-GAT model
class CrossAttentionGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads):
        super(CrossAttentionGAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.gat1 = tg.nn.GATConv(in_dim, hidden_dim, heads=num_heads)
        self.gat2 = tg.nn.GATConv(in_dim, hidden_dim, heads=num_heads)
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.linear = nn.Linear(1024, 128)

    def forward(self, graph1, graph2):
        # Compute initial representations for each graph
        if graph1.edge_index.numel() == 0 or graph1.edge_index.shape[0] == 0\
                or graph2.edge_index.numel() == 0 or graph2.edge_index.shape[0] == 0:
            emb1 = self.proj(graph1.x)
            emb2 = self.proj(graph2.x)

            attention_scores = torch.matmul(emb1, emb2.transpose(0, 1))
            att_weights_1 = torch.softmax(attention_scores, dim=1)
            att_weights_2 = torch.softmax(attention_scores, dim=0)
            cross_att_1 = torch.matmul(att_weights_2, emb2)
            cross_att_2 = torch.matmul(att_weights_1.transpose(0, 1), emb1)

        else:
            emb1 = self.gat1(graph1.x, graph1.edge_index)
            emb2 = self.gat2(graph2.x, graph2.edge_index)

            attention_scores = torch.matmul(emb1, emb2.transpose(0, 1))
            att_weights_1 = torch.softmax(attention_scores, dim=1)
            att_weights_2 = torch.softmax(attention_scores, dim=0)
            cross_att_1 = torch.matmul(att_weights_2, emb2)
            cross_att_2 = torch.matmul(att_weights_1.transpose(0, 1), emb1)\

            cross_att_1 = self.linear(cross_att_1)
            cross_att_2 = self.linear(cross_att_2)

        cross_att_1 = torch.mean(cross_att_1, dim=0)
        cross_att_2 = torch.mean(cross_att_2, dim=0)
        # print(f"cross_att_1: {cross_att_1.shape}, cross_att_2: {cross_att_2.shape}")

        return cross_att_1, cross_att_2


# Define GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, agg_fun):
        super(GraphSAGE, self).__init__()
        self.dropout = nn.Dropout(p = 0.1)
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=agg_fun)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=agg_fun)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        return x
    
# 定义无监督的GraphSAGE损失函数
def unsupervised_loss(z, edge_index):
    adj_scores = torch.mm(z, z.t())  # 邻接矩阵分数
    adj_mask = 1 - torch.eye(adj_scores.shape[0])  # 排除自连接

    # 归一化相似度
    min_value = adj_scores.min()
    max_value = adj_scores.max()
    adj_scores = (adj_scores - min_value) / (max_value - min_value)

    adj_scores = adj_scores * adj_mask  # 仅保留邻居节点之间的分数
    adj_pred = torch.sigmoid(adj_scores)  # 邻接矩阵预测

    # 构建邻接矩阵
    num_nodes = edge_index.max().item() + 1
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    adj_matrix[edge_index[0], edge_index[1]] = 1

    loss = F.binary_cross_entropy(adj_pred, adj_matrix)  # 计算二分类交叉熵损失
    return loss


# Train GraphSAGE model
def graphsage_train(args, input_dim, train_data):
    model = GraphSAGE(input_dim, args.hidden_channels, args.out_channels, args.agg_fun)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader([train_data], batch_size=args.batch_size, shuffle=True)
    print(f"Number of train batches: {len(train_loader)}")

    for epoch in range(args.num_epochs):
        total_loss = 0

        for batch in train_loader:

            optimizer.zero_grad()

            x = model(batch)
            loss = unsupervised_loss(x, batch.edge_index)

            # # Generate positive and negative samples for unsupervised loss
            # pos_edge_index = batch.edge_index
            # neg_edge_index = negative_sampling(batch.edge_index, num_neg_samples=5)
            # loss = unsupervised_loss(x, pos_edge_index, neg_edge_index)

            loss.backward()
            optimizer.step()
            # print(f"batch loss: {loss.item()}")

            total_loss += loss.item()

        # print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {total_loss / len(train_loader)}")

    torch.save(model.state_dict(), args.root + args.cache + 'model_SAGE_' + args.data.split("/")[-2] + args.new_trip + '.pt')
    print("graph model saved.")



# # Use the trained model for inferring vertex reps
# def graphsage_inference(args, input_dim, data, train_class, test_seen_classes, test_unseen_classes):
#     print(f"infer seen classes {len(test_seen_classes)}, unseen classes: {len(test_unseen_classes)}")

#     model = GraphSAGE(input_dim, args.hidden_channels, args.out_channels, args.agg_fun)

#     model.load_state_dict(
#         torch.load(args.root + args.cache + 'model_' + args.g_type + '_' + args.data.split("/")[-2] + args.new_trip + '.pt'))
#     model.eval()

#     pre_embeds = {}
#     out = model(data)
#     for i in range(len(data.y)):
#         pre_embeds[data.y[i]] = out[i]

#     return pre_embeds
