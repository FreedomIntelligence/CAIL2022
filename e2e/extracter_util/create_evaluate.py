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
            src_file = src_path + "result_ensemble2_post2.jsonl"
            self._read_origin_file(src_file)

    def combining(self):
        for sample in tqdm(self.origin_result):
            id = sample["id"]
            text = sample["summary"]
            temp_sample={
                "id":id,
                "text":text
            }
            self.final_result.append(temp_sample)

    def write_result(self, des_file):
        with open(des_file,encoding='utf-8',mode='w') as fw:
            for sample in self.final_result:
                fw.write(json.dumps(sample,ensure_ascii=False)+"\n")


if __name__ == '__main__':
    combiner = Combiner()
    combiner.read_origin_files(src_path="./")
    combiner.combining()
    combiner.write_result(des_file="./test_ensemble016.jsonl")