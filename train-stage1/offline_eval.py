import datetime
import random
import torch
import Training
import numpy as np
from module.XLNet_Encoder import Xlnet_Encoder
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import pandas as pd
import warnings
import argparse

warnings.filterwarnings('ignore')


# 创建子类
class subDataset(Dataset.Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self, Data, Sep, Label):
        self.Data = Data
        self.Sep = Sep
        self.Label = Label

    # 返回数据集大小
    def __len__(self):
        return len(self.Data)

    # 得到数据内容和标签
    def __getitem__(self, index):
        data = torch.LongTensor(self.Data[index])
        sep = torch.LongTensor(self.Sep[index])
        label = torch.LongTensor(self.Label[index])
        return data, sep, label


def load_data(src_path, data_type):
    sen = pd.read_pickle(src_path + data_type + '_sen_tokens.pickle')
    sep = pd.read_pickle(src_path + data_type + '_sep_index.pickle')
    lab = pd.read_pickle(src_path + data_type + '_label.pickle')
    dataset = subDataset(sen, sep, lab)
    if data_type == "train":
        data = DataLoader.DataLoader(dataset, batch_size=1, shuffle=True)
    else:
        data = DataLoader.DataLoader(dataset, batch_size=1, shuffle=False)
    return data


def load_datas(src_path):
    train_data = load_data(src_path=src_path, data_type="train")
    test_data = load_data(src_path=src_path, data_type="test")
    return train_data, test_data


def set_seed():
    np.random.seed(2)
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    random.seed(2)


if __name__ == '__main__':
    import sys

    start = int(sys.argv[1])
    end = int(sys.argv[2])
    set_seed()
    best_model_index_list = [2, 0, 0, 2, 3, 3, 0, 3, 2,3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(start, end):
        model = Xlnet_Encoder(device, model_path="./xlnet-mid").to(device)
        train_data, test_data = load_datas(src_path="ten_fold_data_dir/" + str(i) + "/")
        start_time = datetime.datetime.now()
        print('开始时间：', start_time)
        model_path = "ten_fold_data_dir/" + str(i) + "/"
        Training.get_result(model, test_data, model_path,
                            best_model_index=best_model_index_list[i])  # 观察test情况选择使用哪个epoch
        end_time = datetime.datetime.now()
        print('结束时间：', end_time)
        print('总用时：', end_time - start_time)
