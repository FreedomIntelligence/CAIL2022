import argparse

import torch
from module.model import MT5PForSequenceClassification
from module.tokenizer import T5PegasusTokenizer
from torch.nn.utils.rnn import pad_sequence
import re


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_path', default='./t5_pegasus_torch/', type=str, )
    return parser.parse_args()


class Classifier:
    def __init__(self, vocab_path, model_path, device=0, rs_max_len=200, max_len=512):
        self.tokenizer = T5PegasusTokenizer.from_pretrained(vocab_path)
        args = set_args()
        self.model = MT5PForSequenceClassification(args)
        self.model.load_state_dict(torch.load(model_path))
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.rs_max_len = rs_max_len
        self.max_len = max_len
        self.generate_max_len = rs_max_len

    def convert_feature(self, sample):
        """
            数据处理函数
            Args:
                sample: 一个字典，格式为{"du1": du1, "du2": du2}
            Returns:
            """
        input_ids = []
        text_tokens = self.tokenizer.tokenize(sample["text"])
        # 判断如果正文过长，进行截断
        if len(text_tokens) > self.max_len - self.rs_max_len - 3:
            text_tokens = text_tokens[:self.max_len - self.rs_max_len - 3]
        # 生成模型所需的input_ids和token_type_ids
        input_ids.append(self.tokenizer.cls_token_id)
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(text_tokens))
        input_ids.append(self.tokenizer.sep_token_id)
        return input_ids

    def generate_results(self, output, tokenizer):
        title_list = []
        for i in range(len(output)):
            title = ''.join(tokenizer.decode(output[i][1:], skip_special_tokens=True)).replace(' ', '')
            title_list.append(title)
        return title_list

    def cut_sent(self, para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")

    def quick_predict(self, tex_str):
        sentence_list = self.cut_sent(tex_str)
        summary = "".join(sentence_list[:3])
        return summary

    def predict(self, text_str_list):
        input_list = []
        title_list = []
        input_tensors = []
        for text_str in text_str_list:
            sample = {"text": text_str}
            input_ids = self.convert_feature(sample)
            input_list.append(torch.tensor(input_ids, dtype=torch.long))
        with torch.no_grad():
            input_list = pad_sequence(input_list, batch_first=True, padding_value=0)
            input_tensors = torch.tensor(input_list).long().to(self.device)
            text_output = self.model.generate(input_tensors,
                                              decoder_start_token_id=self.tokenizer.cls_token_id,
                                              eos_token_id=self.tokenizer.sep_token_id,
                                              max_length=self.generate_max_len)
            title_list = self.generate_results(text_output, self.tokenizer)
        return title_list
