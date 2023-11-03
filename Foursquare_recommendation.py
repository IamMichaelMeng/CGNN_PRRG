from tqdm import tqdm
from model.Foursquare.UDA import main as UDA_main
from model.Foursquare.SBGNN import main as SBGNN_main
from model.Foursquare.SRGNN import main as SRGNN_main
from utils.train_print import start_training
from utils.train_print import end_training
from utils.similarity_function import main as similarity_function_main
from utils.model_evaluation import model_evaluate
import torch
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def SBGNN_DATA_to_dict(path):
    # 将用户的SBGNN的数据转换成字典的格式，减少后期的计算时间消耗
    user_check_ins_file = open(path, 'r').readlines()
    user_check_ins_dict = defaultdict(list)
    for eachline in tqdm(user_check_ins_file):
        uid, pid, status = eachline.strip().split()
        uid, pid, status = int(uid), int(pid), int(status)
        user_check_ins_dict[uid].append([uid, pid, status])

    return user_check_ins_dict


def main(num_group, GROUP_SIZE, epoch):
    dataset_name = 'Foursquare'
    # 下面这三个数据集是原数据集自带的，我们需要对其进行处理
    print(f'loading {dataset_name}, user_poi_interaction_data ...')
    poi_interaction_data = SBGNN_DATA_to_dict('datasets/'+dataset_name+'/'+dataset_name+'_user_poi_interaction.txt')
    for group_size in GROUP_SIZE:
        print(f'training {dataset_name} dataset, {group_size} group_size...')
        size_file = 'datasets/' + dataset_name+'/'+dataset_name+'_data_size.txt'

        num_users, num_pois, _ = open(size_file, 'r').readlines()[0].strip('\n').split()
        num_users, num_pois = int(num_users), int(num_pois)

        start_training('UDA')
        UDA_main(num_users, num_pois, dataset_name, group_size, epoch)
        end_training('UDA')
        uda_model = torch.load('datasets/' + dataset_name + '/' + str(group_size) + '_members_group/UDA_model.pt',
                               map_location=device)

        similarity_function_main(uda_model, dataset_name, group_size, poi_interaction_data, device)

        start_training('SBGNN')
        SBGNN_main(dataset_name, group_size, epoch)
        sbgnn_model = torch.load('datasets/' + dataset_name + '/' + str(group_size) + '_members_group/SBGNN_model.pt',
                               map_location=device)
        end_training('SBGNN')

        start_training('SRGNN')
        SRGNN_main(dataset_name, group_size, epoch)
        srgnn_model = torch.load('datasets/' + dataset_name + '/' + str(group_size) + '_members_group/SRGNN_model.pt',
                               map_location=device)
        end_training('SRGNN')

        print('evaluation started...')
        model_evaluate(sbgnn_model, srgnn_model, dataset_name, group_size, num_users, num_pois)


