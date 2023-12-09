#!/usr/bin/env python
# encoding: utf-8
'''
#-------------------------------------------------------------------#
#                   CONFIDENTIAL --- CUSTOM STUDIOS                 #     
#-------------------------------------------------------------------#
#                                                                   #
#                   @Project Name : extracter                 #
#                                                                   #
#                   @File Name    : split.py                      #
#                                                                   #
#                   @Programmer   : Jeffrey                         #
#                                                                   #  
#                   @Start Date   : 2022/9/6 10:06                 #
#                                                                   #
#                   @Last Update  : 2022/9/6 10:06                 #
#                                                                   #
#-------------------------------------------------------------------#
# Classes:                                                          #
#                                                                   #
#-------------------------------------------------------------------#
'''

import json
import random
import os

def set_seed():
    random.seed(2)


class Spliter:
    def __init__(self, num=10):
        self.num_fold = num
        self.sample_list = []
        self.fold_list = [[], [],[],[],[],[],[],[],[],[]]

    def read_data(self, src_file):
        with open(src_file, encoding='utf-8', mode='r') as fr:
            lines = fr.readlines()
            for line in lines:
                sample = json.loads(line)
                self.sample_list.append(sample)
        random.shuffle(self.sample_list)

    def split(self):
        each_num = len(self.sample_list) / self.num_fold
        for index, sample in enumerate(self.sample_list):
            cur_index = int(index // each_num)
            self.fold_list[cur_index].append(sample)

    def save_file(self, des_file, sample_list):
        with open(des_file, encoding='utf-8', mode='a') as fw:
            for sample in sample_list:
                fw.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def save_files(self, save_path):
        for test_index in range(self.num_fold):
            des_path = save_path + "/" + str(test_index) + "/"
            if os.path.exists(des_path):
                pass
            else:
                os.mkdir(des_path)
            for i in range(self.num_fold):
                if i == test_index:
                    des_file = "test.jsonl"
                    self.save_file(des_path + des_file, self.fold_list[i])
                else:
                    des_file = "train.jsonl"
                    self.save_file(des_path + des_file, self.fold_list[i])

if __name__ == '__main__':
    set_seed()
    spliter = Spliter()
    spliter.read_data(src_file="../data_dir/train_out.jsonl")
    spliter.split()
    spliter.save_files(save_path="./")
