from origin_data.converter import create_train_dataset, create_test_dataset
from origin_data.preprocess import Processor
from origin_data.split import set_seed, Spliter

def create_data():
    create_train_dataset()
    create_test_dataset()

def split_data():
    set_seed()
    spliter = Spliter()
    spliter.read_data(src_file="./data_dir/train_out.jsonl")
    spliter.split()
    spliter.save_files(save_path="./ten_fold_data_dir/")

def prepare_data():
    for i in range(10):
        for type in ["train", "test"]:
            processor = Processor(type)
            processor.read_json(src_path="./ten_fold_data_dir/" + str(i) + "/")
            processor.write_file(des_path="./ten_fold_data_dir/" + str(i) + "/")
            if type == "train":
                processor.select_data(token_path="./xlnet-mid", src_path="./ten_fold_data_dir/" + str(i) + "/")
            processor.get_final_data(token_path="./xlnet-mid", src_path="./ten_fold_data_dir/" + str(i) + "/")

if __name__ == '__main__':
    create_data()
    split_data()
    prepare_data()

