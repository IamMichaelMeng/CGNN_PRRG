"""
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)

Modified  on Nov 10, 2017, by Lianhai Miao
Modified  on Nov 15, 2019, by Shuxun Zan
"""
import pdb
import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GDataset(object):
    def __init__(self, group_path, num_negatives, g_m_d):
        """
        Constructor
        """
        self.g_m_d = g_m_d
        self.num_negatives = num_negatives

        # group data，同样的操作对群组的数据再来一遍
        self.group_trainMatrix = self.load_rating_file_as_matrix(group_path + "train.txt", group_path + "test.txt")
        self.group_testRatings = self.load_rating_file_as_list(group_path + "test.txt")
        self.group_testNegatives = self.load_negative_file(group_path + "test.txt")
        self.num_users, self.num_items = self.group_trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        rating_list = []
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])
                rating_list.append([user, item])
                line = f.readline()
        return rating_list

    def load_negative_file(self, filename):
        test_items = []
        for line in open(filename, 'r'):
            # 把测试集中的所有item的id提取出来放在一个list中
            contents = line.split(' ')
            test_item_id = int(contents[1])
            test_items.append(test_item_id)

        # 下面这一步操作着实不理解作者的目的是在干嘛
        negativeList = []
        for line in open(filename, 'r'):
            negativeList.append(test_items)
        return negativeList

    def load_rating_file_as_matrix(self, filename, test_filename):
        # Get number of users and items
        num_users, num_items = 0, 0
        # 下面这两个for循环是为了计算训练集和测试集中用户和item的数量分别是多少
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split(" ")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()

        with open(test_filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split(" ")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()

        # Construct matrix
        # sp.dok_matrix采用字典来记录矩阵中的非零元素，字典中的key对应的是矩阵中非零元素的下标，value则是记录元素特有的数值
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)  # mat[994,1683]

        # 构建<用户，item>交互矩阵，如果用户和item产生过交互，那么就将矩阵对应位置元素设为1.0，否则设为0
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split(" ")
                if len(arr) > 2:
                    # 如果用户有评分记录
                    user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if rating > 0:
                        mat[user, item] = 1.0
                else:
                    # 如果用户没有评分记录
                    user, item = int(arr[0]), int(arr[1])
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    def get_train_instances(self, train):
        # train = self.user_trainMatrix
        user_input, pos_item_input, neg_item_input = [], [], []
        num_users = train.shape[0]
        num_items = train.shape[1]
        for (u, i) in train.keys():
            # (u,i) == (user_id, item_id)，如果处理的是group数据的话，这里的u指的是group_id
            # 一个(u,i)记录对应5个正样本，5个负样本，但是正样本全部都是一样的，这是为了和后面的负样本进行配对。样本是用户没有交互过的item，也可能存在重复负样本id
            # positive instance
            for _ in range(self.num_negatives):
                pos_item_input.append(i)
            # negative instances
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)  # 每一条记录给扩展成5份
                neg_item_input.append(j)
        pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]  # 一个正样本和一个负样本封装起来
        return user_input, pi_ni  # 一个u跟着一个一组[正样本item,负样本item]

    def get_user_dataloader(self, batch_size):
        user, positem_negitem_at_u = self.get_train_instances(self.user_trainMatrix)
        train_data = TensorDataset(torch.tensor(user).to(device), torch.tensor(user).to(device),
                                   torch.tensor(positem_negitem_at_u).to(device))
        user_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return user_train_loader

    def get_group_dataloader(self, batch_size):
        group, positem_negitem_at_g = self.get_train_instances(self.group_trainMatrix)
        group_members = []
        for gid in group:
            group_members.append(self.g_m_d[gid])  # 把每一条签到记录的群组成员给加进来，如果num_negative的数量很多，且群组成员很多，那么这里的group_members数组会很大很大，很耗内存

        # 封装成tensor类型
        train_data = TensorDataset(torch.tensor(group).to(device), torch.tensor(group_members).to(device), torch.tensor(positem_negitem_at_g).to(device))

        # 按照batch_size对数据进行划分
        group_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return group_train_loader
