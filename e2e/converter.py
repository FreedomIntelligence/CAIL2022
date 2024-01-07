#!/usr/bin/env python
# encoding: utf-8
'''
#-------------------------------------------------------------------#
#                   CONFIDENTIAL --- CUSTOM STUDIOS                 #     
#-------------------------------------------------------------------#
#                                                                   #
#                   @Project Name : t5-segment                 #
#                                                                   #
#                   @File Name    : converter.py                      #
#                                                                   #
#                   @Programmer   : Jeffrey                         #
#                                                                   #  
#                   @Start Date   : 2022/8/14 10:09                 #
#                                                                   #
#                   @Last Update  : 2022/8/14 10:09                 #
#                                                                   #
#-------------------------------------------------------------------#
# Classes:                                                          #
#                                                                   #
#-------------------------------------------------------------------#
'''
from rouge import Rouge
from sample import Sample
import json
from tqdm import tqdm
import re
import sys

sys.setrecursionlimit(10000000)

class Converter:
    def __init__(self, sample):
        self.id = sample["id"]
        self.summary = sample["summary"]
        self.text = sample["text"]
        self.summary_sent_list = []
        self.text_sent_list = []
        self.text_list = []
        self.sample_list = []
        self.score_matrix = []

    def cut_sent(self, para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")

    def segment_sentences(self):
        self.summary_sent_list = self.cut_sent(self.summary)
        self.text_sent_list = self.cut_sent(self.text)

    def _eval_rouge_score(self, source, target, rouge_eval):
        source, target = ' '.join(source), ' '.join(target)
        try:
            scores = rouge_eval.get_scores(hyps=source, refs=target)
            score = {
                'rouge-1': scores[0]['rouge-1']['f'],
                'rouge-2': scores[0]['rouge-2']['f'],
                'rouge-l': scores[0]['rouge-l']['f'],
            }
        except ValueError:
            score = {
                'rouge-1': 0.0,
                'rouge-2': 0.0,
                'rouge-l': 0.0,
            }
        final_score = 0.2 * score["rouge-1"] + 0.3 * score["rouge-2"] + 0.5 * score["rouge-l"]
        return final_score

    def create_zero_text_list(self):
        for i in range(0, len(self.text_sent_list)):
            self.text_list.append({"sentence": self.text_sent_list[i], "important": 0})

    def _create_score_matrix(self):
        for i in range(len(self.text_sent_list)):
            temp_summary_score = []
            for j in range(len(self.summary_sent_list)):
                temp_summary_score.append(0)
            self.score_matrix.append(temp_summary_score)

    def _find_most_important_x_y(self):
        highest_score = 0
        highest_row = -1
        highest_col = -1
        for row in range(len(self.text_sent_list)):
            for col in range(len(self.summary_sent_list)):
                if self.score_matrix[row][col] > highest_score:
                    highest_score = self.score_matrix[row][col]
                    highest_row = row
                    highest_col = col
        return highest_row, highest_col

    def _set_zero(self, row, col):
        for i in range(len(self.text_sent_list)):
            self.score_matrix[i][col] = 0
        for j in range(len(self.summary_sent_list)):
            self.score_matrix[row][j] = 0

    def _find_most_important_index(self):
        for i in range(len(self.summary_sent_list)):
            row, col = self._find_most_important_x_y()
            self._set_zero(row, col)
            self.text_list[row]["important"] = col + 1

    def label_important_sentence(self, rouge_eval):
        self._create_score_matrix()
        for row, sentence in enumerate(self.text_sent_list):
            for col, summary in enumerate(self.summary_sent_list):
                final_score = self._eval_rouge_score(summary, sentence, rouge_eval)
                self.score_matrix[row][col] = final_score
        self._find_most_important_index()

    def get_samples(self):
        # sample = {"id": 0, "segment": 0, "text": [{"sentence": "sentence1", "important": "1"}],"summary":[]}
        sample = Sample(self.id, 0)
        sample.label(self.text_list, self.summary_sent_list)
        self.sample_list.append(sample)

class ConverterForTest:
    def __init__(self, sample):
        self.id = sample["id"]
        self.text = sample["text"]
        self.text_sent_list = []
        self.sample_list = []

    def cut_sent(self, para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")

    def segment_sentences(self):
        self.text_sent_list = self.cut_sent(self.text)

    def get_samples(self, ):
        sample = Sample(self.id, 0)
        sample.label_for_test(self.text_sent_list)
        self.sample_list.append(sample)

def create_test_dataset():
    data_type = "evaluate"
    input_file = "./data_dir/" + data_type + ".jsonl"
    output_file = "./data_dir/" + data_type + "_out.jsonl"
    with open(input_file, encoding="utf-8", mode='r') as fr:
        lines = fr.readlines()
        for step, line in enumerate(tqdm(lines)):
            origin_sample = json.loads(line)
            converter = ConverterForTest(origin_sample)
            converter.segment_sentences()
            converter.get_samples()
            sample_list = converter.sample_list
            with open(output_file, encoding='utf-8', mode='a') as fw:
                for sample in sample_list:
                    fw.write(json.dumps(sample.to_json(), ensure_ascii=False) + '\n')
if __name__ == '__main__':
    create_test_dataset()
