# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import pdb
import sys
import os
# import pdb
import torch
import argparse
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

# from torch.utils.tensorboard import SummaryWriter
sys.path.append('../')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--dirpath', default=BASE_DIR, help='Current Dir')
parser.add_argument('--dataset_name', type=str, default='Yelp')
parser.add_argument('--a_emb_size', type=int, default=9, help='Embeding Size')
parser.add_argument('--b_emb_size', type=int, default=9, help='Embeding Size')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight_Decay')
parser.add_argument('--lr', type=float, default=0.005, help='Learning Rate')
parser.add_argument('--seed', type=int, default=13, help='Random seed')
parser.add_argument('--epoch', type=int, default=1, help='Epoch')
parser.add_argument('--gnn_layer_num', type=int, default=2, help='GNN Layer')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch Size')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout')
parser.add_argument('--agg', type=str, default='AttentionAggregator', choices=['AttentionAggregator', 'MeanAggregator'],
                    help='Aggregator')
args = parser.parse_args()

# TODO
# args.device = 'cpu'
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

exclude_hyper_params = ['dirpath', 'device']
hyper_params = dict(vars(args))
for exclude_p in exclude_hyper_params:
    del hyper_params[exclude_p]
hyper_params = "~".join([f"{k}-{v}" for k, v in hyper_params.items()])


# tb_writer = SummaryWriter(comment=hyper_params)


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup seed


setup_seed(args.seed)

this_fpath = os.path.abspath(__file__)
DATA_EMB_DIC1 = {'Yelp': (30887, 18995)}
DATA_EMB_DIC = {**DATA_EMB_DIC1}
for k in DATA_EMB_DIC1:
    for i in range(1, 6):
        DATA_EMB_DIC.update({
            f'{k}-{i}': DATA_EMB_DIC1[k]})


class AttentionAggregator(nn.Module):
    def __init__(self, a_dim, b_dim):
        super(AttentionAggregator, self).__init__()
        self.out_mlp_layer = nn.Sequential(
            nn.Linear(b_dim, b_dim),
        )

        self.a = nn.Parameter(torch.FloatTensor(a_dim + b_dim, 1))
        nn.init.kaiming_normal_(self.a.data)

    def forward(self, edge_dic_list: dict, feature_a, feature_b, node_num_a, node_num_b, matrix, sign):

        edges = []
        for node in range(node_num_a):
            neighs = np.array(edge_dic_list[node]).reshape(-1, 1)
            a = np.array([node]).repeat(len(neighs)).reshape(-1, 1)
            edges.append(np.concatenate([a, neighs], axis=1))

        edges = np.vstack(edges)
        edges = torch.LongTensor(edges).to(args.device)
        new_emb = feature_b
        new_emb = self.out_mlp_layer(new_emb)

        # 下面开始处理权重
        if sign is False:
            # 如果是FaLase，表示当前聚合时不需要使用权重
            edge_h_2 = torch.cat([feature_a[edges[:, 0]], new_emb[edges[:, 1]]],
                                 dim=1)  # 这里就是将两组feature直接联结起来，什么操作也没有施加,简单粗暴就直接连接上了,如果使用权重的话就用在这里
        else:
            edge_h_2 = []
            for edge in edges:
                tem_fea_a = feature_a[edge[0]].reshape(1, -1)
                tem_fea_b = new_emb[edge[1]].reshape(1, -1)
                weight = matrix[edge[0]][edge[1]]
                tem_fea_b = weight * tem_fea_b  # 在测试是可以构建全1矩阵
                edge_h_2.append(torch.cat([tem_fea_a, tem_fea_b], dim=1))
            edge_h_2 = torch.vstack(edge_h_2)  # 再把列表转换成矩阵

        edges_h = torch.exp(F.elu(torch.einsum("ij,jl->il", [edge_h_2, self.a]), 0.1))
        matrix = torch.sparse_coo_tensor(edges.t(), edges_h[:, 0], torch.Size([node_num_a, node_num_b]),
                                         device=args.device)
        row_sum = torch.sparse.mm(matrix, torch.ones(size=(node_num_b, 1)).to(args.device))
        row_sum = torch.where(row_sum == 0, torch.ones(row_sum.shape).to(args.device), row_sum)

        output_emb = torch.sparse.mm(matrix, new_emb)
        output_emb = output_emb.div(row_sum)
        return output_emb


class SBGNNLayer(nn.Module):
    def __init__(self, num_users, num_pois, edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,
                 edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg,
                 dataset_name, emb_size_a=9, emb_size_b=9, aggregator=AttentionAggregator):
        super(SBGNNLayer, self).__init__()
        self.set_a_num, self.set_b_num = num_users, num_pois
        self.edgelist_a_b_pos, self.edgelist_a_b_neg, self.edgelist_b_a_pos, self.edgelist_b_a_neg = \
            edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg
        self.edgelist_a_a_pos, self.edgelist_a_a_neg, self.edgelist_b_b_pos, self.edgelist_b_b_neg = \
            edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg

        self.agg_a_from_b_pos = aggregator(emb_size_b, emb_size_a)
        self.agg_a_from_b_neg = aggregator(emb_size_b, emb_size_a)
        self.agg_a_from_a_pos = aggregator(emb_size_a, emb_size_a)
        self.agg_a_from_a_neg = aggregator(emb_size_a, emb_size_a)

        self.agg_b_from_a_pos = aggregator(emb_size_a, emb_size_b)
        self.agg_b_from_a_neg = aggregator(emb_size_a, emb_size_b)
        self.agg_b_from_b_pos = aggregator(emb_size_b, emb_size_b)
        self.agg_b_from_b_neg = aggregator(emb_size_b, emb_size_b)

        self.update_func = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(emb_size_a * 5, emb_size_a * 2),
            nn.PReLU(),
            nn.Linear(emb_size_b * 2, emb_size_b)
        )

    def forward(self, feature_a, feature_b, matrix):
        node_num_a, node_num_b = self.set_a_num, self.set_b_num
        m_a_from_b_pos = self.agg_a_from_b_pos(self.edgelist_a_b_pos, feature_a, feature_b, node_num_a, node_num_b,
                                               matrix, True)
        m_a_from_b_neg = self.agg_a_from_b_neg(self.edgelist_a_b_neg, feature_a, feature_b, node_num_a, node_num_b,
                                               matrix, False)
        m_a_from_a_pos = self.agg_a_from_a_pos(self.edgelist_a_a_pos, feature_a, feature_a, node_num_a, node_num_a,
                                               matrix, False)
        m_a_from_a_neg = self.agg_a_from_a_neg(self.edgelist_a_a_neg, feature_a, feature_a, node_num_a, node_num_a,
                                               matrix, False)
        new_feature_a = torch.cat([feature_a, m_a_from_b_pos, m_a_from_b_neg, m_a_from_a_pos, m_a_from_a_neg], dim=1)
        new_feature_a = self.update_func(new_feature_a)
        m_b_from_a_pos = self.agg_b_from_a_pos(self.edgelist_b_a_pos, feature_b, feature_a, node_num_b, node_num_a,
                                               matrix.T, True)
        m_b_from_a_neg = self.agg_b_from_a_neg(self.edgelist_b_a_neg, feature_b, feature_a, node_num_b, node_num_a,
                                               matrix, False)
        m_b_from_b_pos = self.agg_b_from_b_pos(self.edgelist_b_b_pos, feature_b, feature_b, node_num_b, node_num_b,
                                               matrix, False)
        m_b_from_b_neg = self.agg_b_from_b_neg(self.edgelist_b_b_neg, feature_b, feature_b, node_num_b, node_num_b,
                                               matrix, False)
        new_feature_b = torch.cat([feature_b, m_b_from_a_pos, m_b_from_a_neg, m_b_from_b_pos, m_b_from_b_neg], dim=1)
        new_feature_b = self.update_func(new_feature_b)

        return new_feature_a, new_feature_b


class SBGNN(nn.Module):
    def __init__(self, edgelists, dataset_name, num_users, num_pois,
                 layer_num=1, emb_size_a=9, emb_size_b=9, aggregator=AttentionAggregator):
        super(SBGNN, self).__init__()
        assert len(edgelists) == 8, 'must 8 edgelists'
        edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg, edgelist_a_a_pos, edgelist_a_a_neg, \
        edgelist_b_b_pos, edgelist_b_b_neg = edgelists
        self.set_a_num, self.set_b_num = num_users, num_pois

        # 这俩是user和item的特征向量
        self.features_a = nn.Embedding(self.set_a_num, emb_size_a)
        self.features_b = nn.Embedding(self.set_b_num, emb_size_b)
        self.features_a.weight.requires_grad = True
        self.features_b.weight.requires_grad = True

        self.layers = nn.ModuleList(
            [SBGNNLayer(num_users, num_pois, edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos,
                        edgelist_b_a_neg, edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos,
                        edgelist_b_b_neg, dataset_name=args.dataset_name, emb_size_a=9, emb_size_b=9,
                        aggregator=aggregator) for _ in range(layer_num)]
        )

    def get_embeddings(self, matrix):
        # 拿到user和POI的特征向量表示
        matrix = matrix.to(args.device)
        emb_a = self.features_a(torch.arange(self.set_a_num).to(args.device))
        emb_b = self.features_b(torch.arange(self.set_b_num).to(args.device))

        # 将两组特征向量表示送入神经网络层执行前向传播
        for m in self.layers:
            emb_a, emb_b = m(emb_a, emb_b, matrix)
        return emb_a, emb_b

    def forward(self, edge_lists, matrix):
        embedding_a, embedding_b = self.get_embeddings(matrix)
        if embedding_a.device == 'cuda' and embedding_b.device == 'cuda':
            embedding_a, embedding_b = embedding_a.cpu(), embedding_b.cpu()
        if torch.is_tensor(edge_lists):
            edge_lists = edge_lists.numpy()
        y = torch.einsum("ij, ij->i", [embedding_a[edge_lists[:, 0]], embedding_b[edge_lists[:, 1]]])
        return torch.sigmoid(y), embedding_a, embedding_b

    def loss(self, pred_y, y):
        assert y.min() >= 0, 'must 0~1'
        assert pred_y.size() == y.size(), 'must be same length'
        # pos_ratio = y.sum() / y.size()[0]
        # weight = torch.where(y > 0.5, 1./pos_ratio, 1./(1-pos_ratio))
        return F.binary_cross_entropy(pred_y, y, weight=None)


def forward_for_model(model, edge_lists):
    embedding_a, embedding_b = model.get_embeddings()
    user_embedding = embedding_a[edge_lists[0]]
    poi_embedding = embedding_b[edge_lists[1]]
    user_embedding = user_embedding.reshape(1, user_embedding.shape[0])
    poi_embedding = poi_embedding.reshape(1, poi_embedding.shape[0])

    return user_embedding, poi_embedding


# =========== function
def load_data(dataset_name, group_size):
    train_edgelist = np.loadtxt('datasets/' + dataset_name + '/' + str(group_size) + '_members_group/' + dataset_name +
                                '_user_poi_interaction.txt', dtype=np.int32)
    train_edgelist = train_edgelist[0:100]
    return train_edgelist


# ============= load data
def load_edgelists(edge_lists):
    edgelist_a_b_pos, edgelist_a_b_neg = defaultdict(list), defaultdict(list)
    edgelist_b_a_pos, edgelist_b_a_neg = defaultdict(list), defaultdict(list)
    edgelist_a_a_pos, edgelist_a_a_neg = defaultdict(list), defaultdict(list)
    edgelist_b_b_pos, edgelist_b_b_neg = defaultdict(list), defaultdict(list)
    for a, b, s in edge_lists:
        # pdb.set_trace()
        if s > 1 or s == 1:
            # 这里需要对原始SBGNN的代码进行更改，由s==1改为s>1 or s==1
            edgelist_a_b_pos[a].append(b)
            edgelist_b_a_pos[b].append(a)
        elif s == -1:
            edgelist_a_b_neg[a].append(b)
            edgelist_b_a_neg[b].append(a)
        else:
            # print(a, b, s)
            raise Exception("s must be -1/1")

    edge_list_a_a = defaultdict(lambda: defaultdict(int))
    edge_list_b_b = defaultdict(lambda: defaultdict(int))
    for a, b, s in edge_lists:
        for b2 in edgelist_a_b_pos[a]:
            edge_list_b_b[b][b2] += 1 * s
        for b2 in edgelist_a_b_neg[a]:
            edge_list_b_b[b][b2] -= 1 * s
        for a2 in edgelist_b_a_pos[b]:
            edge_list_a_a[a][a2] += 1 * s
        for a2 in edgelist_b_a_neg[b]:
            edge_list_a_a[a][a2] -= 1 * s

    for a1 in edge_list_a_a:
        for a2 in edge_list_a_a[a1]:
            v = edge_list_a_a[a1][a2]
            if a1 == a2:
                continue
            if v > 0:
                edgelist_a_a_pos[a1].append(a2)
            elif v < 0:
                edgelist_a_a_neg[a1].append(a2)

    for b1 in edge_list_b_b:
        for b2 in edge_list_b_b[b1]:
            v = edge_list_b_b[b1][b2]
            if b1 == b2:
                continue
            if v > 0:
                edgelist_b_b_pos[b1].append(b2)
            elif v < 0:
                edgelist_b_b_neg[b1].append(b2)

    return edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg, \
           edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg


def create_matrix(edge_lists, set_a_num, set_b_num):
    # 根据边集合构建user与item交互矩阵，矩阵元素表示交互次数
    matrix = torch.zeros((set_a_num, set_b_num))
    for edge in edge_lists:
        # 先构建训练集的矩阵
        if edge[2] == 1:
            matrix[edge[0], edge[1]] += 1

    normalized_matrix = F.normalize(matrix, dim=1)
    return normalized_matrix  # 返回更新的矩阵


def run(args, dataset_name, group_size):
    train_edgelist = load_data(dataset_name, group_size)
    set_a_num, set_b_num = DATA_EMB_DIC[args.dataset_name]
    train_matrix = create_matrix(train_edgelist, set_a_num, set_b_num)
    train_y = np.array([i[-1] for i in train_edgelist])
    train_y = torch.from_numpy((train_y + 1) / 2).float().to(args.device)
    # get edge lists
    edgelists = load_edgelists(train_edgelist)

    agg = AttentionAggregator

    model = SBGNN(edgelists, args.dataset_name, set_a_num, set_b_num, args.gnn_layer_num, aggregator=agg)
    model = model.to(args.device)

    # print(model.train())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = 1000.0
    best_embed = []
    for epoch in tqdm(range(0, args.epoch)):
        # train
        model.train()
        optimizer.zero_grad()
        pred_y, embedding_a, embedding_b = model(train_edgelist, train_matrix)
        loss = model.loss(pred_y, train_y)
        loss.backward()
        optimizer.step()

        # res_cur = {}
        best_model = model
        if loss < best_loss:
            best_model = model
            best_embed.clear()  # 只保留最新的数据
            best_embed.append([embedding_a, embedding_b])
    return best_model, best_embed[0]


def main(dataset_name, group_size, epoch):
    args.epoch = epoch
    best_model, embedding = run(args, dataset_name, group_size)
    torch.save(best_model, 'datasets/' + dataset_name + '/' + str(group_size) + '_members_group/SBGNN_model.pt')
    embedding_a, embedding_b = embedding[0], embedding[1]
    return embedding_a, embedding_b
