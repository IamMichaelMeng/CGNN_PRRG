'''
分别使用欧几里得、余弦相似度、皮尔森系数三种方法
寻找虚拟用户的相似用户
'''
import random

import torch
import pdb
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os


def takeSecond(elem):
    return elem[1]


def load_user_check_ins(dataset_name):
    # 获取用户的签到数据
    user_check_ins_file = open('datasets/' + dataset_name + '/' + dataset_name + '_check_ins.txt', 'r').readlines()
    user_check_ins_dict = defaultdict(set)
    for eachline in tqdm(user_check_ins_file):
        uid, pid, _ = eachline.strip().split()
        uid, pid = int(uid), int(pid)
        user_check_ins_dict[uid].add(pid)

    return user_check_ins_dict


def cosine_similarity(groups_embed, users_feature, group_size, data_name, device):
    # 寻找与群组特征相似的相似用户
    similar_user_id_list = []
    for group_id in range(0, groups_embed.shape[0]):
        # 挨个群组进行相似度的计算
        cosine_similarity_list = []
        group_emb = groups_embed[group_id]
        group_emb = group_emb.cpu().detach().numpy()
        for user_idx, user_info in enumerate(tqdm(users_feature)):
            user_info = user_info.reshape(1, -1)[0][0: 9]
            user_info = user_info.cpu().detach().numpy()
            num = float(np.dot(user_info, group_emb.T))  # dot
            denom = np.linalg.norm(user_info) * np.linalg.norm(group_emb)  # multiplile
            similarity = 0.5 + 0.5 * (num / denom) if denom != 0 else 0
            cosine_similarity_list.append([user_idx, similarity])

        cosine_similarity_list.sort(key=takeSecond, reverse=True)
        # 只取前20名相似用户的IDn
        cosine_similarity_list = np.array(cosine_similarity_list)[:20]
        similar_users_id = list(cosine_similarity_list[:, 0])
        similar_users_id.insert(0, group_id)
        similar_user_id_list.append(similar_users_id)
        # 将结果保存起来
    np.savetxt('datasets/' + data_name + '/' + str(group_size) + '_members_group/' + data_name + '_cosine_dis_set.txt',
               np.array(similar_user_id_list), fmt='%d')
    return similar_user_id_list


def create_group_check_ins(similar_user_list, dataset_name, group_size):
    # 将相似用户的历史签到记录作为群组的签到记录，这部分数据是用来测试模型最终的性能的
    old_check_ins = load_user_check_ins(dataset_name)
    poi_ids = []
    print(f'procesing {dataset_name} dataset, {group_size} group_size for SRGNN_data...')
    for group_id in range(0, len(similar_user_list)):
        similar_users = similar_user_list[group_id][1:]
        similar_users = [int(x) for x in similar_users]  # 转为int类型的数据
        check_in_set = set()
        for user_id in similar_users:
            # 对于每一个相似用户
            user_check_ins = old_check_ins[user_id]
            user_check_ins = list(user_check_ins)
            sorted(user_check_ins)
            for poi in user_check_ins:
                if poi not in check_in_set:
                    check_in_set.add(poi)

        check_in_list = list(check_in_set)[:20]
        temp = sorted(check_in_list)
        temp.insert(0, group_id)
        poi_ids.append(temp)
    # 保存数据
    np.savetxt('datasets/' + dataset_name + '/' + str(
        group_size) + '_members_group/' + dataset_name + '_single_positive_group_poi.txt', np.array(poi_ids), fmt='%d',
               delimiter=' ')


def SRGNN_process(similar_user_array, dataset_name, group_size):
    # 将相似用户的历史签到记录作为SRGNN模型的训练数据和整体模型的测试数据
    old_check_ins = load_user_check_ins(dataset_name)

    print(f'procesing {dataset_name} dataset, {group_size} group_size for SRGNN_data...')
    SRGNN_data = []
    overall_test_data = []
    for group_id in range(0, len(similar_user_array)):
        similar_users = similar_user_array[group_id][1:]
        similar_users = [int(x) for x in similar_users]  # 转为int类型的数据
        # 对于每一个相似用户，将一个相似用户的签到数据一部分拿出来用作训练集，一部分用作整体模型的测试
        for user_id in similar_users:
            pois = list(old_check_ins[user_id])
            random.seed(10)
            random.shuffle(pois)
            srgnn_train = pois[:int(len(pois) * 0.6)]
            overall_test = pois[int(len(pois) * 0.6):]
            sorted(srgnn_train)
            sorted(overall_test)
            for poi in srgnn_train:
                SRGNN_data.append([user_id, poi, 1])
            for poi in overall_test:
                overall_test_data.append([group_id, poi])
    # 保存数据
    np.savetxt('datasets/' + dataset_name + '/' + str(group_size) + '_members_group/' + dataset_name +
               '_user_poi_transition.txt', np.array(SRGNN_data), fmt='%d', delimiter=' ')
    np.savetxt(
        'datasets/' + dataset_name + '/' + str(group_size) + '_members_group/' + dataset_name +
        '_single_positive_group_poi.txt', np.array(overall_test_data), fmt='%d', delimiter=' ')


def SBGNN_process(similar_user_list, dataset_name, group_size, poi_interaction_data):
    # 处理SBGNN的数据集,将相似用户签到数据作为SBGNN模型的训练数据
    old_sbgnn_data = poi_interaction_data
    print(f'processing {dataset_name}, {group_size} group_size dataset for SBGNN_data...')
    file = open('datasets/' + dataset_name + '/' + str(group_size) + '_members_group/' + dataset_name +
                '_user_poi_interaction.txt', mode='w')
    for group_id in range(0, len(similar_user_list)):
        # 挨个群组进行处理
        similar_users = similar_user_list[group_id][1:]
        similar_users = [int(x) for x in similar_users]  # 转为int类型的数据
        # 对于每一个相似用户
        for user_id in similar_users:
            data = old_sbgnn_data[user_id]  # 获取相似用户的poi签到记录
            for elem in data:
                file.writelines([str(elem[0]) + ' ', str(elem[1]) + ' ', str(elem[2]) + ' ', '\n'])
    file.close()


def main(model, data_name, group_size, poi_interaction_data, device):
    # 加载idx_map文件

    # 读取模型数据
    print(f'processing {data_name} dataset, {group_size} group_size...')
    # model = torch.load('datasets/' + data_name + '/' + str(group_size) + '_members_group/UDA_model.pt', map_location='cpu')  # 读取一个组中的group以及user的特征表示
    g_embs = model.groupembeds.weight.data
    groups_emb = model.mapping_group_embeds(g_embs)
    u_embeds = model.userembeds.weight.data
    users_feature = model.mapping_group_member_embeds(u_embeds)

    # 寻找相似用户
    cosine_similarity_list = cosine_similarity(groups_emb, users_feature, group_size, data_name, device)
    # 基于相似用户构建二部图的训练集和测试集
    SBGNN_process(cosine_similarity_list, data_name, group_size, poi_interaction_data)
    # 基于相似用户构建转移图的训练集和测试集
    SRGNN_process(cosine_similarity_list, data_name, group_size)
    # 构建群组的历史交互数据用以验证模型的性能
    create_group_check_ins(cosine_similarity_list, data_name, group_size)
