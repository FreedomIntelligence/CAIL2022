import datetime
import random
import torch
import Training
from combine import Combiner
from decoder_result import Decoder
from module.XLNet_Encoder import Xlnet_Encoder
from offline_eval import set_seed, load_datas
from postprocess import postprocess1


def get_label():
    start = 0
    end = 10
    set_seed()
    #best_model_index_list = [2, 0, 0, 2, 3, 3, 0, 3, 2, 3]
    best_model_index_list = [1, 3, 2, 1, 0, 3, 3, 1, 3, 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(start, end):
        model = Xlnet_Encoder(device, model_path="./xlnet-mid").to(device)
        train_data, test_data = load_datas(src_path="ten_fold_data_dir/" + str(i) + "/")
        start_time = datetime.datetime.now()
        print('开始时间：', start_time)
        model_path = "ten_fold_data_dir/" + str(i) + "/"
        Training.get_result(model, test_data, model_path,
                            best_model_index=best_model_index_list[i])  # 观察test情况选择使用哪个epoch
        end_time = datetime.datetime.now()
        print('结束时间：', end_time)
        print('总用时：', end_time - start_time)
def get_result():
    root = "ten_fold_data_dir/"
    for i in range(10):
        id_file = root + str(i) + "/" + "test.jsonl"
        text_file = root + str(i) + "/" + "test_txt.pickle"
        label_file = root + str(i) + "/" + "label_out.pickle"
        des_file = root + "result_" + str(i) + ".jsonl"
        decoder = Decoder()
        decoder.load_data(id_file, text_file, label_file)
        decoder.convert(des_file)
def combine():
    combiner = Combiner()
    combiner.read_origin_files(src_path="./ten_fold_data_dir/")
    combiner.read_summary_file(src_file="./origin_data/train.jsonl")
    combiner.combining()
    combiner.write_result(des_file="./train_temp.jsonl")

if __name__ == '__main__':
    get_label()
    get_result()
    combine()
    postprocess1()