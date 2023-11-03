import pdb

from tqdm import tqdm
from collections import defaultdict
# import pdb
import numpy as np
import random
import json


def load_user_check_ins():
    # 获取用户的签到数据
    user_check_ins_file = open('../datasets/'+d_name+'/'+d_name+'_check_ins.txt', 'r').readlines()
    user_check_ins_dict = defaultdict(set)
    for each_line in tqdm(user_check_ins_file):
        uid, pid, _ = each_line.strip().split()
        uid, pid = int(uid), int(pid)
        user_check_ins_dict[uid].add(pid)

    return user_check_ins_dict


def create_group_rating(d_name, group_member_list, group_size):
    """
    生成群组的rating文件，这个数据集是为了给UDA模型构建训练、测试数据
    1 获取群组成员的历史签到记录
    2 将群组成员的历史签到记录作为群组的历史签到记录
    3 保存群组的历史签到记录
    """
    user_check_ins = load_user_check_ins()
    group_check_ins = defaultdict(set)
    for group in group_member_list:
        group_id = group[0]
        group_members = group[1:]
        for member_id in group_members:
            group_check_ins[group_id].update(user_check_ins[member_id])  # 将群组成员的访问过的poi作为群组的poi
        # 建立文件
        group_rating_data = []
        for key in group_check_ins.keys():
            pois = group_check_ins[key]
            for poi in pois:
                group_rating_data.append([key, poi, 1])  # 这里的key是群组的id
    np.savetxt('../datasets/'+d_name+'/'+str(group_size)+'_members_group/group_check_ins.txt', np.array(group_rating_data), fmt='%d', delimiter=' ')
    return group_check_ins


def create_different_size_group_member(data_size_path, num_group, group_size, file, node_idx_map_file):
    """
    1 获取用户数量、打乱数据
    2 按照群组尺寸对用户进行分组，不满一组的从其他组随机挑选用户
    3 保存群组成员数据
    4 构建群组的check-ins数据
    5 保存群组的check-ins数据
    """
    data_size = np.loadtxt(data_size_path, dtype=np.int32)
    all_members = list(set(range(data_size[0])))
    group_member_list, node_idx_map_list = [], []
    for index in range(0, num_group):
        group_member = random.sample(all_members, group_size)  # 随机选择指定数量的群组成员
        node_idx_map = {j: i for i, j in enumerate(group_member)}  # 先构建出新的user_id
        node_idx_map_list.append(node_idx_map)
        # 给群组加上群组id
        group_member.insert(0, index)
        group_member_list.append(group_member)
        # 保存群组成员文件
        for elem in group_member[:-1]:
            file.write(str(elem) + ' ')  # 先写入前n-1个
        file.write(str(group_member[-1])+' '+'\n')  # 再写入最后一个，并追加一个换行符
        json.dump(node_idx_map, node_idx_map_file)
        node_idx_map_file.write('\n')
    return group_member_list, node_idx_map_list


def split_data(d_name, group_size, group_check_ins):
    # 划分UDA模型的训练集和测试集合
    random.seed(10)  # 设定随机数的种子
    group_check_ins_list = []
    for key in group_check_ins.keys():
        pois = group_check_ins[key]
        for poi in pois:
            group_check_ins_list.append([key, poi, 1])
    # 打乱数据
    random.seed(10)
    random.shuffle(group_check_ins_list)
    # 划分训练集、测试集
    group_rating_train_array = group_check_ins_list[:int(len(group_check_ins_list)*0.7)]
    group_rating_test_array = group_check_ins_list[int(len(group_check_ins_list)*0.7):]

    # 保存文件
    np.savetxt('../datasets/'+d_name+'/'+str(group_size)+'_members_group/group_train.txt', group_rating_train_array,
               fmt='%d')
    np.savetxt('../datasets/'+d_name+'/'+str(group_size)+'_members_group/group_test.txt', group_rating_test_array,
               fmt='%d')


def process_data(d_name, data_size_path, num_group, group_size, file, node_idx_map_file):
    # 先划分群组
    group_member_list, node_idx_map_list = create_different_size_group_member(data_size_path, num_group, group_size, file, node_idx_map_file)
    # 生成group_rating文件，这个文件是给UDA和SRGNN两个模型使用的
    group_check_ins = create_group_rating(d_name, group_member_list, group_size)
    # 划分uda模型使用的训练集和测试集
    split_data(d_name, group_size, group_check_ins)


if __name__ == '__main__':
    """
    大概的处理思路如下：
        1 根据事先指定的群组的尺寸将用户的数据集随机划分成若干个随机群组
        2 群组划分后，每一个群组都是相互独立的一个个体，需要对群组成员的id进行重新映射，否则UDA模型会报错数组越界的错误
        3 对sbgnn数据集进行重新映射
        4 对srgnn数据集进行重新映射
    所有的id重映射只需要对用户进行即可，因为poi从始至终就没有发生变化
    """
    dataset_name = ['Foursquare', 'Gowalla', 'Yelp']
    dataset_size = {'Foursquare': [2551, 13474], 'Gowalla': [5628, 31803], 'Yelp': [30887, 18995]}
    group_size = 20  # 群组规模
    num_group = 60  # 群组数量
    GROUP_SIZE = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for d_name in dataset_name:
        for group_size in GROUP_SIZE:
            print(f'processing {d_name} dataset...')
            data_size = np.loadtxt('../datasets/'+d_name+'/'+d_name+'_data_size.txt', dtype=np.int32)
            data_size_path = '../datasets/'+d_name+'/'+d_name+'_data_size.txt'
            file = open('../datasets/' + d_name + '/' + str(group_size) + '_members_group/group_members.txt', 'w')
            node_idx_map_file = open(
                '../datasets/' + d_name + '/' + str(group_size) + '_members_group/' + d_name + '_node_idx_map.json', 'w')
            print('processing train group...')
            process_data(d_name, data_size_path, num_group, group_size, file, node_idx_map_file)
            file.close()
            node_idx_map_file.close()
