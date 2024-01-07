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
            if len(summary) < 50:
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
        self.delete_report()  # 删除前奏的报道
        self.delete_repeat()  # 删除重复的句子
        self.delete_titles()  # 删除标题
        self.delete_start()  # 删除开始的前奏报道

    def cut_sent(self, para):
        para = re.sub('([*。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")

    def _delete_repeat_sentence(self, sentence_list):
        new_summary = ""
        for sentence in sentence_list:
            if sentence in new_summary:
                continue
            new_summary += sentence
        return new_summary

    def delete_repeat(self):
        for i in range(len(self.result_sample_list)):
            sample = self.result_sample_list[i]
            summary = sample["summary"]
            temp_sentence_list = self.cut_sent(summary)
            summary = self._delete_repeat_sentence(temp_sentence_list)
            self.result_sample_list[i]["summary"] = summary

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

    def delete_report(self):
        for i in range(len(self.result_sample_list)):
            sample = self.result_sample_list[i]
            summary = sample["summary"]
            index = summary.find("》报道：")
            if index != -1:
                summary = summary[index + 4:]

            self.result_sample_list[i]["summary"] = summary
    def _delete(self, strs):
        if strs == '':
            return strs
        else:
            str = re.sub(u"\<.*?\>|\(.*?\)|\{.*?\}|\[.*?\]|\（.*?\）|\【.*?\】", '', strs)
            str = re.sub('\<|\>|\(|\)|\{|\}|\[|\]|\（|\）|\【|\】', '', str)
        str = str.replace("原标题：", "")
        return str

    def delete_titles(self):
        for i in range(len(self.result_sample_list)):
            sample = self.result_sample_list[i]
            summary = sample["summary"]
            summary = self._delete(summary)
            self.result_sample_list[i]["summary"] = summary

    def pair_bracket(self, strs):
        all_bracket = []
        for item in strs:
            if item == '《':
                all_bracket.append(item)
            elif item == '》':
                all_bracket.append(item)
            else:
                continue
        flag = 0
        for i in all_bracket:
            if i == '《':
                flag += 1
            elif i == '》':
                flag -= 1
            if flag < 0:
                return False
        if len(all_bracket) % 2 == 0:
            return True
        else:
            return False

    def start_delect(self, summary):
        """
         input  summary_list:['***', '***', ..., '***']
         output  all_summary:['***', '***', ..., '***']
        """
        all_summary = ""
        if re.match('.*?发布\《', summary) and len(re.match('.*?发布\《', summary).group()) <= 25:
            if self.pair_bracket(summary) == False:
                summary = re.sub('.*?发布\《', '', summary)
                all_summary = summary
            else:
                all_summary = summary
        else:
            all_summary = summary
        return all_summary

    def delete_start(self):
        for i in range(len(self.result_sample_list)):
            sample = self.result_sample_list[i]
            summary = sample["summary"]
            summary = self.start_delect(summary)
            self.result_sample_list[i]["summary"] = summary

def postprocess2():
    postpreprocess = Postpreprocess()
    postpreprocess.read_origin_file("./data_dir/evaluate_out.jsonl")
    postpreprocess.read_result_file("./output/result_temp.jsonl")
    postpreprocess.pipeline_process()
    postpreprocess.save_file("./result.jsonl")

if __name__ == '__main__':
    postpreprocess = Postpreprocess()
    postpreprocess.read_origin_file("../data_dir/evaluate_out.jsonl")
    postpreprocess.read_result_file("result126-3_200.jsonl")
    postpreprocess.pipeline_process()
    postpreprocess.save_file("result126-3_post.jsonl")
