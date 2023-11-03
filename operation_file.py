# 主要是用于文件的创建和删除，拒绝双手操作
import os


def delete_file(file_path):
    file_name = [d_name+'_cosine_dis_set.txt', d_name+'_single_positive_group_poi.txt', 'UDA_model.pt']
    # file_name = [d_name+'_GAT_trained_user_features.txt', d_name+'_group_emb.txt']
    for f_name in file_name:
        print(f'deleting {d_name} dataset, {g_size} group_size, {f_name} file...')
        try:
            os.remove(file_path + f_name)
        except Exception:
            continue


if __name__ == '__main__':
    dataset_name = ['Foursquare']
    group_size = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for d_name in dataset_name:
        for g_size in group_size:
            delete_file('./datasets/'+d_name+'/'+str(g_size)+'_members_group/')
