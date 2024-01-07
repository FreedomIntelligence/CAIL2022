from tqdm import tqdm
import json
import re


class Postpreprocess:
    def __init__(self):
        self.origin_sample_list = []
        self.result_sample_list = []

    def read_origin_file(self, src_file):
        with open(src_file, encoding="utf-8", mode='r') as fr:
            lines = fr.readlines()
            for step, line in enumerate(tqdm(lines)):
                origin_sample = json.loads(line)
                self.origin_sample_list.append(origin_sample)

    def read_result_file(self, src_file):
        with open(src_file, encoding="utf-8", mode='r') as fr:
            lines = fr.readlines()
            for step, line in enumerate(tqdm(lines)):
                result_sample = json.loads(line)
                self.result_sample_list.append(result_sample)

    def save_file(self, des_file):
        with open(des_file, encoding='utf-8', mode='w') as fw:
            for sample in self.result_sample_list:
                fw.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def fill_zero_sentence(self):
        for i in range(len(self.result_sample_list)):
            sample = self.result_sample_list[i]
            summary = sample["summary"]
            if len(summary) ==0:
                self.result_sample_list[i]["summary"] = self._get_start_sentence(sample["id"])

    def _find(self, string):
        url = re.findall('www(?:[-\w.]|(?:%[\da-fA-F]{2}))+', string)
        return url

    def _find2(self, string):
        url = re.findall('http://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', string)
        return url

    def delete_url(self):
        for i in range(len(self.result_sample_list)):
            sample = self.result_sample_list[i]
            summary = sample["summary"]
            urls = self._find(summary)
            if len(urls) != 0:
                for url in urls:
                    summary = summary.replace(url, "")
            urls = self._find2(summary)
            if len(urls) != 0:
                for url in urls:
                    summary = summary.replace(url, "")
            self.result_sample_list[i]["summary"] = summary

    def _check_parentheses(self, summary):  # 可以全部删除
        matches = re.findall("<.*?>", summary)
        return matches

    def _check_parentheses2(self, summary):  # 这个可以删
        matches = re.findall("【.*?】", summary)
        return matches

    def pipeline_process(self):
        self.fill_zero_sentence()  # 先填充没有抽取出的结果，按照前三句抽取。
        self.delete_url()  # 删除摘要中的URL，对其他步骤没有影响
        self.replace_special_character()  # 删除摘要中的特殊空白字符，对其他步骤没有影响
        self.delete_brackets()  # 删除括号

    def _get_start_sentence(self, id, num=3):
        summary = ""
        for sample in self.origin_sample_list:
            if sample["id"] == id:
                for i in range(num):
                    if i < len(sample["text"]):
                        summary += sample["text"][i]["sentence"]
                return summary

    def replace_special_character(self):
        for i in range(len(self.result_sample_list)):
            sample = self.result_sample_list[i]
            summary = sample["summary"]
            summary = summary.replace("​", "")
            summary = summary.replace(" ", "")
            summary = summary.replace("■", "")
            summary = summary.replace("&;nbsp", "")
            summary = summary.replace("●", "")
            summary = summary.replace(" ", "")
            summary = summary.replace("（）", "")
            summary = summary.replace("()", "")
            summary = summary.replace("&nbsp", "")
            self.result_sample_list[i]["summary"] = summary

    def delete_brackets(self):
        for i in range(len(self.result_sample_list)):
            sample = self.result_sample_list[i]
            summary = sample["summary"]
            matches = self._check_parentheses(summary)
            if len(matches) != 0:
                for match in matches:
                    summary = summary.replace(match, "")
            matches = self._check_parentheses2(summary)
            if len(matches) != 0:
                for match in matches:
                    summary = summary.replace(match, "")
            self.result_sample_list[i]["summary"] = summary

def postprocess1():
    postpreprocess = Postpreprocess()
    postpreprocess.read_origin_file("./data_dir/evaluate_out.jsonl")
    postpreprocess.read_result_file("./data_dir/result_ensemble.jsonl")
    postpreprocess.pipeline_process()
    postpreprocess.save_file("./data_dir/result_ensemble_post.jsonl")

if __name__ == '__main__':
    postpreprocess = Postpreprocess()
    postpreprocess.read_origin_file("../data_dir/evaluate_out.jsonl")
    postpreprocess.read_result_file("../data_dir/result_ensemble.jsonl")
    postpreprocess.pipeline_process()
    postpreprocess.save_file("../data_dir/result_ensemble_post.jsonl")
