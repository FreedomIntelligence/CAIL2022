import json

from tqdm import tqdm


class Combiner:
    def __init__(self):
        self.origin_result = []
        self.new_result = []
        self.final_result = []

    def _read_origin_file(self, src_file):
        with open(src_file, encoding='utf-8', mode='r') as fr:
            lines = fr.readlines()
            for line in lines:
                example = json.loads(line)
                self.origin_result.append(example)

    def read_origin_files(self, src_path):
        for i in range(10):
            src_file = src_path + "result_" + str(i) + ".jsonl"
            self._read_origin_file(src_file)

    def read_summary_file(self, src_file):
        with open(src_file, encoding='utf-8', mode='r') as fr:
            lines = fr.readlines()
            for line in lines:
                example = json.loads(line)
                self.new_result.append(example)

    def _find_summary(self, ids):
        for sample in self.new_result:
            if sample["id"] == ids:
                return sample["summary"]

    def combining(self):
        for sample in tqdm(self.origin_result):
            id = sample["id"]
            text = sample["summary"]
            summary = self._find_summary(id)
            temp_sample={
                "id":id,
                "text":text,
                "summary":summary
            }
            self.final_result.append(temp_sample)

    def write_result(self, des_file):
        with open(des_file,encoding='utf-8',mode='w') as fw:
            for sample in self.final_result:
                fw.write(json.dumps(sample,ensure_ascii=False)+"\n")


if __name__ == '__main__':
    combiner = Combiner()
    combiner.read_origin_files(src_path="./ten_fold_data_dir/")
    combiner.read_summary_file(src_file="./origin_data/train.jsonl")
    combiner.combining()
    combiner.write_result(des_file="./train.jsonl")