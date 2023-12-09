import pandas as pd
import json


class Decoder:
    def __init__(self):
        self.eid = []
        self.etxt = []
        self.elab = []

    def _get_id_file(self, ori_file):
        all_ids = []
        with open(ori_file, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                dict = json.loads(line)
                all_ids.append(dict['id'])
        return all_ids

    def load_data(self, id_file, text_file, label_file):
        self.eid = self._get_id_file(id_file)
        self.etxt = pd.read_pickle(text_file)
        self.elab = pd.read_pickle(label_file)

    def convert(self, des_file):
        with open(des_file, 'a', encoding='utf-8') as f:
            for i, j, k in zip(self.eid, self.etxt, self.elab):
                str = ''
                for index, item in enumerate(k):
                    if item == 1:
                        str += j[index]
                temp_dict = {"id": i, "summary": str}
                a = json.dumps(temp_dict, ensure_ascii=False)
                f.writelines(a + '\n')


if __name__ == '__main__':
    root = "ten_fold_data_dir/"
    for i in range(10):
        id_file = root + str(i) + "/" + "test.jsonl"
        text_file = root + str(i) + "/" + "test_txt.pickle"
        label_file = root + str(i) + "/" + "label_out.pickle"
        des_file = root+"result_"+str(i)+".jsonl"
        decoder = Decoder()
        decoder.load_data(id_file, text_file, label_file)
        decoder.convert(des_file)
