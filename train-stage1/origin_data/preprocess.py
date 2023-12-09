#!/usr/bin/env python
# encoding: utf-8
'''
#-------------------------------------------------------------------#
#                   CONFIDENTIAL --- CUSTOM STUDIOS                 #     
#-------------------------------------------------------------------#
#                                                                   #
#                   @Project Name : extracter                 #
#                                                                   #
#                   @File Name    : preprocess.py                      #
#                                                                   #
#                   @Programmer   : Jeffrey                         #
#                                                                   #  
#                   @Start Date   : 2022/9/6 11:25                 #
#                                                                   #
#                   @Last Update  : 2022/9/6 11:25                 #
#                                                                   #
#-------------------------------------------------------------------#
# Classes:                                                          #
#                                                                   #
#-------------------------------------------------------------------#
'''
import json
import pickle
from transformers import AutoTokenizer
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class Processor:
    def __init__(self, type):
        self.all_txt = []
        self.all_label = []
        self.type = type
        self.train_max_len = 2400
        self.test_max_len = 6000

    def read_json(self, src_path):
        with open(src_path + "/" + self.type + ".jsonl", 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                dict = json.loads(line)
                txt, label, lab = [], [], []
                for item in dict['text']:
                    txt.append(item['sentence'])
                    label.append(item['important'])
                    if item['important'] == 0:
                        lab.append(0)
                    else:
                        lab.append(1)
                self.all_txt.append(txt)
                self.all_label.append(lab)

    def write_file(self, des_path):
        if self.type=="train":
            with open(des_path + '/' + self.type + '_txt_original.pickle', 'wb') as f:
                string = pickle.dumps(self.all_txt)
                f.write(string)
            with open(des_path + '/' + self.type + '_label_original.pickle', 'wb') as f:
                string = pickle.dumps(self.all_label)
                f.write(string)
        else:
            with open(des_path + '/' + self.type + '_txt.pickle', 'wb') as f:
                string = pickle.dumps(self.all_txt)
                f.write(string)
            with open(des_path + '/' + self.type + '_label.pickle', 'wb') as f:
                string = pickle.dumps(self.all_label)
                f.write(string)

    def select_data(self, token_path, src_path):
        sen_ = pd.read_pickle(src_path + 'train_txt_original.pickle')
        lab_ = pd.read_pickle(src_path + 'train_label_original.pickle')
        tokenizer = AutoTokenizer.from_pretrained(token_path)
        want_sen = []
        want_lab = []
        for sen, lab in zip(sen_, lab_):
            output = []
            for sentence in sen:
                token = tokenizer.tokenize(sentence)
                token = token + ['<sep>']
                output += token
            output.append('<cls>')
            token_id = tokenizer.convert_tokens_to_ids(output)
            if len(token_id) > self.train_max_len * 2:
                continue
            elif len(token_id) > self.train_max_len:
                rate = self.train_max_len / len(token_id)
                new_sen = []
                for sentence in sen:
                    lens = int(len(sentence) * rate) - 1
                    new_sen.append(sentence[:lens])
                want_sen.append(new_sen)
                want_lab.append(lab)
            else:
                want_sen.append(sen)
                want_lab.append(lab)
        print(len(want_lab), len(want_sen))
        with open(src_path + 'train_txt.pickle', 'wb') as f:
            string = pickle.dumps(want_sen)
            f.write(string)
        with open(src_path + 'train_label.pickle', 'wb') as f:
            string = pickle.dumps(want_lab)
            f.write(string)

    def get_final_data(self, token_path, src_path):
        tokenizer = AutoTokenizer.from_pretrained(token_path)
        sens = pd.read_pickle(src_path + '/'+self.type+'_txt.pickle')
        all_token = []
        all_sep = []
        max_len = 0
        if self.type=="train":
            max_len=self.train_max_len
        else:
            max_len=self.test_max_len
        for index, sen in enumerate(sens):
            output = []
            sep_index = []
            for sentence in sen:
                token = tokenizer.tokenize(sentence)
                token = token + ['<sep>']
                output += token
                sep_index.append(len(output) - 1)
            output.append('<cls>')
            token_id = tokenizer.convert_tokens_to_ids(output)
            if len(token_id) > max_len:
                rate = max_len / len(token_id)
                output1 = []
                sep_index1 = []
                for sentence in sen:
                    lens = int(len(sentence) * rate) - 1
                    token = tokenizer.tokenize(sentence[:lens])
                    token = token + ['<sep>']
                    output1 += token
                    sep_index1.append(len(output1) - 1)
                output1.append('<cls>')
                token_id1 = tokenizer.convert_tokens_to_ids(output1)
                sep_index = sep_index1
            else:
                token_id1 = token_id
            all_token.append(token_id1)
            all_sep.append(sep_index)
        with open(src_path + self.type+'_sen_tokens.pickle', 'wb') as f:
            string = pickle.dumps(all_token)
            f.write(string)
        with open(src_path + self.type+'_sep_index.pickle', 'wb') as f:
            string = pickle.dumps(all_sep)
            f.write(string)


if __name__ == '__main__':
    for i in range(10):
        for type in ["train", "test"]:
            processor = Processor(type)
            processor.read_json(src_path="../ten_fold_data_dir/" + str(i) + "/")
            processor.write_file(des_path="../ten_fold_data_dir/" + str(i) + "/")
            if type == "train":
                processor.select_data(token_path="../xlnet-mid", src_path="../ten_fold_data_dir/" + str(i) + "/")
            processor.get_final_data(token_path="../xlnet-mid", src_path="../ten_fold_data_dir/" + str(i) + "/")
