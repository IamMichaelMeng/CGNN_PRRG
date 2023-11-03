# import pdb
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils.uda_util import Helper
from utils.uda_dataset import GDataset
import pdb
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class UDA(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio):
        super(UDA, self).__init__()
        # 下面这三行代码是随机初始化数据对象的特征向量，模拟数据对象自身携带部分元信息
        self.userembeds = nn.Embedding(num_users, 64)
        self.itemembeds = nn.Embedding(num_items, 32)
        self.groupembeds = nn.Embedding(num_groups, 18)
        # 下面三行是对初始向量进行预处理，将不同维度的输入向量映射至统一的向量空间中
        self.mapping_group_member_embeds = GroupMemberEmbeddingLayer(64, 30, embedding_dim)
        self.mapping_item_embeds = ItemEmbeddingLayer(32, 16, embedding_dim)
        self.mapping_group_embeds = GroupEmbeddingLayer(18, 10, embedding_dim)

        self.attention = AttentionLayer(embedding_dim, drop_ratio)
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)
        self.group_member_dict = group_member_dict
        self.num_users = num_users
        self.num_groups = len(self.group_member_dict)
        random_gid = random.sample(list(self.group_member_dict.keys()), 1)
        self.group_size = len(self.group_member_dict[random_gid[0]])
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, group_inputs, item_inputs, group_members, sign):
        out = self.grp_forward(group_inputs, item_inputs, group_members, sign)
        return out

    def grp_forward(self, group_inputs, item_inputs, group_members, sign):
        # 三组初始特征
        initial_group_member_embeds = self.userembeds(group_members)
        initial_item_embeds = self.itemembeds(item_inputs)
        # initial_group_embeds = self.groupembeds(group_inputs)
        # 三组映射后的特征
        mapped_group_member_embeds = self.mapping_group_member_embeds(initial_group_member_embeds)
        mapped_item_embeds = self.mapping_item_embeds(initial_item_embeds)
        # mapped_group_embeds = self.mapping_group_embeds(initial_group_embeds)
        '''
            item_inputs是poi的id，下面会根据id来从itemembeds中提取对应的item的嵌入表示
        '''
        item_embeds_full = mapped_item_embeds  # [batch_size, embedding_size]
        '''
        [batch_size, group_size, embedding_size], 每一个群组成员的embeds都提取出来了，包括0号用户，这时masks就能分辨出哪些是真用户，哪些是填数用的僵尸用户
        '''
        members_embeds = mapped_group_member_embeds
        members_embeds = torch.squeeze(members_embeds, dim=1)
        item_embeds = mapped_item_embeds  # [batch_size, embedding_size]

        at_wt = self.attention(members_embeds, item_embeds)  # [batch_size, group_size]  一个组中的所有成员针对同一个item都要计算一个权重
        # 下面的代码就是自己对UDA模型改进的代码, 如果修改的话就从这里开始修改为原来的代码
        temp_at_wt = torch.zeros(at_wt.shape[0], at_wt.shape[1]).to(device)
        for idx, group_id in enumerate(group_inputs):
            # 对每一个组,由原来的两两比较变成随机选定一个用户做为标杆进行比较
            randid = np.random.randint(3)  # 随机选择一个用户
            rand_score = at_wt[idx][randid]  # 再选择这个随机用户的权重作为标杆
            rand_score = rand_score.expand(1, at_wt.shape[1])  # 维度扩充，便于下一步的计算
            # new_score = (at_wt[idx] - rand_score).clone().detach().requires_grad_(True)
            temp_at_wt[idx] = at_wt[idx] - rand_score  # 所有用户的权重都和标杆用户的权重做减法
        at_wt = temp_at_wt
        at_wt = torch.abs(at_wt)

        # 下面这行代码是原始的UDA模型的代码
        at_wt = at_wt.reshape(at_wt.shape[0], 1, self.group_size)

        # 下面两步操作就是加权融合得到群组表示
        g_embeds_with_attention = torch.bmm(at_wt, members_embeds)
        g_embeds_with_attention = torch.squeeze(g_embeds_with_attention)
        g_embeds = g_embeds_with_attention
        if sign == 'train':
            element_embeds = torch.mul(g_embeds, item_embeds_full)  # element-wise product
            new_embeds = torch.cat((element_embeds, g_embeds, item_embeds_full), dim=1)
            y = torch.sigmoid(self.predictlayer(new_embeds))
            return y
        else:
            pdb.set_trace()
            g_embeds_with_attention = torch.mean(g_embeds_with_attention, dim=0, keepdim=True)
            return g_embeds_with_attention, item_embeds_full


class GroupMemberEmbeddingLayer(nn.Module):
    def __init__(self, user_dim, hidden_dim, out_put_dim):
        super(GroupMemberEmbeddingLayer, self).__init__()
        self.user_mapping_layer = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_put_dim)
        )

    def forward(self, user_input):
        user_mapped = self.user_mapping_layer(user_input)
        return user_mapped


class GroupEmbeddingLayer(nn.Module):
    def __init__(self, group_dim, hidden_dim, out_put_dim):
        super(GroupEmbeddingLayer, self).__init__()
        self.group_mapping_layer = nn.Sequential(
            nn.Linear(group_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_put_dim)
        )

    def forward(self, group_input):
        group_mapped = self.group_mapping_layer(group_input)
        return group_mapped


class ItemEmbeddingLayer(nn.Module):
    def __init__(self, item_dim, hidden_dim, out_put_dim):
        super(ItemEmbeddingLayer, self).__init__()
        self.item_mapping_layer = nn.Sequential(
            nn.Linear(item_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_put_dim)
        )

    def forward(self, item_input):
        item_mapped = self.item_mapping_layer(item_input)
        return item_mapped


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )

    def forward(self, members_embeds, item_embeds):
        difference = []
        for i in range(members_embeds.shape[1]):
            ui = members_embeds[:, i]
            i_difference = []
            for j in range(members_embeds.shape[1]):
                if i == j:
                    continue
                uj = members_embeds[:, j]
                i_difference.append((ui - uj) * item_embeds)

            i_difference = torch.stack(i_difference)
            i_difference = torch.sum(i_difference, 0)
            difference.append(i_difference)

        outputs = []
        for i in range(len(difference)):
            out = self.linear(difference[i])
            outputs.append(out)
        outputs = torch.cat(outputs, dim=1)
        weights = torch.softmax(outputs, dim=1)
        return weights


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out


def parse_args():
    parser = argparse.ArgumentParser(description="Run UDA.")
    parser.add_argument('--embedding_size', type=int, default=9, help='embedding_size')
    parser.add_argument('--epoch', type=int, default=1, help='Number of epochs, the default is 32.')
    parser.add_argument('--num_negatives', type=int, default=1, help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--batch_size', type=int, default=1024, help='default batch size is 1024, but the GPU memory is limited.Batch size.')
    parser.add_argument('--drop_ratio', type=float, default=0.3, help='')
    parser.add_argument('--lr', nargs='?', default='[0.01, 0.005, 0.001]', help="lr.")
    return parser.parse_args()


# train the model
def training(model, train_loader, epoch_id, args):
    # user training
    learning_rates = args.lr

    # learning rate decay
    lr = learning_rates[0]
    if 15 <= epoch_id < 25:
        lr = learning_rates[1]
    elif epoch_id >= 20:
        lr = learning_rates[2]
    # lr decay
    if epoch_id % 5 == 0:
        lr /= 2

    # optimizer
    optimizer = optim.RMSprop(model.parameters(), lr)
    epoch_loss = 0
    for batch_id, (u, group_members, pi_ni) in enumerate(train_loader):
        # Data Load
        group_input = u  # group_ids,长度是一个batch
        pos_item_input = pi_ni[:, 0]  # 群组交互的正样本
        neg_item_input = pi_ni[:, 1]  # 群组交互的负样本
        # Forward
        pos_prediction = model(group_input, pos_item_input, group_members, 'train')
        neg_prediction = model(group_input, neg_item_input, group_members, 'train')
        # Zero_grad
        model.zero_grad()
        # Loss
        loss = torch.mean((pos_prediction - neg_prediction - 1) ** 2)
        epoch_loss += loss.data.cpu().numpy()

        # Backward
        loss.backward(retain_graph=True)
        optimizer.step()

    return epoch_loss


def main(num_users, num_items, dataset_name, group_size, epoch):
    args = parse_args()
    args.epoch = epoch
    args.lr = eval(args.lr)
    args.path = 'datasets/'+dataset_name+'/'+str(group_size)+'_members_group/'
    args.group_dataset = args.path + 'group_'  # 这个路径是train和test两个数据集使用的
    args.user_in_group_path = 'datasets/' + dataset_name + '/'+str(group_size)+'_members_group'+'/group_members.txt'

    # initial helper
    helper = Helper()

    # get the dict of users in group
    g_m_d = helper.gen_group_member_dict(args.user_in_group_path)
    # initial dataSet class
    dataset = GDataset(args.group_dataset, args.num_negatives, g_m_d)
    # get group number
    num_group = max(g_m_d.keys()) + 1
    num_users = num_users
    num_items = num_items
    # build UDA model
    agree = UDA(num_users, num_items, num_group, args.embedding_size, g_m_d, args.drop_ratio).to(device)
    best_loss = 1000.0
    best_model = agree
    for epoch in tqdm(range(args.epoch)):
        # training
        agree.train()
        train_loader = dataset.get_group_dataloader(128)
        loss = training(agree, train_loader, epoch, args)
        if loss <= best_loss:
            best_model = agree

    # 将最好的模型保存至本地
    # return best_model
    torch.save(best_model, 'datasets/'+dataset_name+'/'+str(group_size)+'_members_group/UDA_model.pt')
