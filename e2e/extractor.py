import torch
import pandas as pd
import json
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
from module.XLNet_Encoder import Xlnet_Encoder
from extracter_util.preprocess import Processor
from tqdm import tqdm


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


class Classifier:
    def __init__(self,ensemble=True):
        self.ensemble = ensemble
        self.eid = []
        self.etxt = []
        self.elab = [[], [], []]

    def _get_id_file(self, ori_file):
        all_ids = []
        with open(ori_file, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                dict = json.loads(line)
                all_ids.append(dict['id'])
        return all_ids

    def load_id_txt(self, id_file, text_file):
        self.eid = self._get_id_file(id_file)
        self.etxt = pd.read_pickle(text_file)

    def get_result(self, model, evaluate_data, model_path, best_model_index, model_index):
        model.load_state_dict(torch.load(model_path + '/saved_model/epoch{num}.ckpt'.format(num=best_model_index)))
        model.eval()
        predict_all = []
        with torch.no_grad():
            for item in tqdm(evaluate_data):
                sentence = item[0].cuda()
                sep = item[1].cuda()
                class_out = model(sentence, sep)
                predict = torch.max(class_out.data, 1)[1].cpu().numpy()
                predict_all.append(predict)

        self.elab[model_index] = predict_all

    def convert(self, des_file):
        with open(des_file, 'a', encoding='utf-8') as f:
            print("Ensemble:",self.ensemble)
            for i, j, k0, k1, k2 in tqdm(zip(self.eid, self.etxt, self.elab[0], self.elab[1], self.elab[2])):
                str = ''
                if self.ensemble:
                    for index, item0 in enumerate(k0):
                        item1 = k1[index]
                        item2 = k2[index]
                        item = item0 + item1 + item2
                        if item > 1:
                            str += j[index]
                else:
                    for index, item0 in enumerate(k0):
                        if item0 == 1:
                            str += j[index]

                str = str.replace("\t", "")
                temp_dict = {"id": i, "summary": str}
                a = json.dumps(temp_dict, ensure_ascii=False)
                f.writelines(a + '\n')

    def load_data(self, data_type):
        sen = pd.read_pickle("data_dir/" + data_type + '_sen_tokens.pickle')
        sep = pd.read_pickle("data_dir/" + data_type + '_sep_index.pickle')
        lab = pd.read_pickle("data_dir/" + data_type + '_label.pickle')
        dataset = subDataset(sen, sep, lab)
        if data_type == "train":
            data = DataLoader.DataLoader(dataset, batch_size=1, shuffle=True)
        else:
            data = DataLoader.DataLoader(dataset, batch_size=1, shuffle=False)
        return data

def preprocess():
    processor = Processor("evaluate")
    processor.read_json(src_path="./data_dir/")
    processor.write_file(des_path="./data_dir/")
    processor.get_final_data(token_path="./xlnet-mid", src_path="./data_dir/")

def extract(ensemble = True):
    best_model_index_list = [2, 0, 0, 2, 3, 3, 0, 3, 2, 3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id_file = "data_dir/evaluate.jsonl"
    text_file = "data_dir/evaluate_txt.pickle"
    des_file = "data_dir/result_ensemble.jsonl"
    model = Xlnet_Encoder(device, model_path="./xlnet-mid").to(device)
    classifier = Classifier(ensemble = ensemble)
    evaluate_data = classifier.load_data(data_type="evaluate")
    for index, i in enumerate([1, 2, 6]):
        model_path = "extractor_model/" + str(i) + "/"
        classifier.get_result(model, evaluate_data, model_path, best_model_index_list[i], index)
    classifier.load_id_txt(id_file, text_file)
    classifier.convert(des_file)
