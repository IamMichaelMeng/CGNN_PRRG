# 模型的测试函数
import pdb
import random
from collections import defaultdict
import math
from utils.SRGNN_utils import Data
from model.Foursquare.SRGNN import test as srgnn_test
from model.Foursquare.SBGNN import create_matrix
from tqdm import tqdm
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def precision_recall_ndcg(predicted_items, true_items, k):
    # 计算推荐物品与真实喜欢的物品的交集
    intersection = set(predicted_items[:k]).intersection(set(true_items))
    # 计算准确率
    precision = len(intersection) / k
    # 计算召回率
    recall = len(intersection) / len(true_items)
    # 计算 NDCG
    dcg = sum([(1 if item in intersection else 0) / math.log2(i + 2) for i, item in enumerate(predicted_items[:k])])
    idcg = sum([1 / math.log2(i + 2) for i in range(min(k, len(true_items)))])
    ndcg = dcg / idcg
    # 返回结果
    return precision, recall, ndcg


def load_similar_user_id(dataset_name, group_size):
    file = open('datasets/'+dataset_name+'/'+str(group_size)+'_members_group/'+'group_members.txt', 'r').readlines()
    similar_user_id_dict = defaultdict(list)
    for each_line in file:
        arr = each_line.strip().split(' ')
        g_id = int(arr[0])
        similar_users = arr[1:]
        similar_users = [int(x) for x in similar_users]
        similar_user_id_dict[g_id].append(similar_users)
    return similar_user_id_dict


def load_poi_test(dataset_name, group_size):
    file = open('datasets/' + dataset_name + '/' + str(group_size) + '_members_group/' +
                                 dataset_name + '_single_positive_group_poi.txt', 'r').readlines()
    pois = set()
    for each_line in file:
        arr = each_line.strip().split(' ')
        pois.add(int(arr[1]))
    return list(pois)


def load_group_member(dataset_name, group_size):
    # 加载群组成员
    file = open('datasets/' + dataset_name + '/' + str(group_size) + '_members_group/'
                + 'group_members.txt').readlines()
    group_members = defaultdict(list)
    for each_line in file:
        arr = each_line.strip().split(' ')
        arr = [int(x) for x in arr]
        group_id = arr[0]
        members = arr[1:]
        group_members[group_id].append(members)
    return group_members


def load_similar_user_poi_interaction_data(dataset_name, similar_user_id_list, num_users, num_pois):
    # 获取相似用户的POI交互数据
    similar_user_poi_interaction = []
    data_name = ['training', 'validation', 'testing']
    for d_name in data_name:
        # print(f'loading similar users poi interaction data from {d_name}...')
        user_poi_interaction = np.loadtxt('datasets/' + dataset_name + '/' + dataset_name + '_'+d_name+'.txt',
                                          dtype=np.int)
        for item in user_poi_interaction:
            if item[0] in similar_user_id_list and item[1] != num_pois:
                similar_user_poi_interaction.append(item)
    tempfile = np.array(similar_user_poi_interaction)
    # print('creating similar users poi interaction matrix...')
    matrix = create_matrix(tempfile, num_users, num_pois)
    test_y = np.array([i[-1] for i in tempfile])
    similar_user_poi_interaction = torch.from_numpy(tempfile)
    return similar_user_poi_interaction, matrix, test_y


def load_similar_user_poi_transition_data(dataset_name, group_size, similar_user_id_list):
    # 获取相似用户的POI转移数据
    user_poi_transition_data = np.loadtxt('datasets/' + dataset_name + '/' + dataset_name +
                      '_check_ins.txt', dtype=np.int)
    similar_user_poi_transition_dict = defaultdict(list)
    for elem in user_poi_transition_data:
        if elem[0] in similar_user_id_list and elem[1] not in similar_user_poi_transition_dict[elem[0]]:
            similar_user_poi_transition_dict[elem[0]].append(elem[1])
    user_poi_check_data = list(similar_user_poi_transition_dict.values())
    target_data = []
    for poi_sequence in user_poi_check_data:
        target_data.append(poi_sequence[-1])
        del poi_sequence[-1]  # 再把最后一个元素在原序列中删除掉
    similar_user_poi_transition_data = (user_poi_check_data, target_data)
    similar_user_poi_transition_data = Data(similar_user_poi_transition_data, shuffle=True)
    return similar_user_poi_transition_data, target_data


def model_evaluate(sbgnn_model, srgnn_model, dataset_name, group_size, num_users, num_pois):
    # 模型性能评估
    group_check_ins = np.loadtxt('datasets/' + dataset_name + '/' + str(group_size) + '_members_group/' +
                                 dataset_name + '_single_positive_group_poi.txt', dtype=np.int32)
    k = int(group_check_ins.shape[0]*0.5)  # 从测试集中随机选择一半
    # 使用 random.choice() 函数随机选择 k 行
    random_rows = np.random.choice(group_check_ins.shape[0], size=k, replace=False)
    # 根据随机选择的行索引获取对应的行
    group_test_check_ins = group_check_ins[random_rows]
    similar_user_id_dict = load_similar_user_id(dataset_name, group_size)
    model_performance = [[[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]]
    for i in range(0, len(model_performance)):
        for j in range(0, len(model_performance[i])):
            mean_performance[i].append(np.mean(model_performance[i][j]))
    np.savetxt('results/' + dataset_name + '_' + str(group_size) + '_model_performance.txt', np.array(mean_performance), fmt='%f')
    print('evaluation ended.')

