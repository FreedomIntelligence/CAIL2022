import json
import random
import numpy as np
import torch
import os
from classfier import Classifier
from tqdm import tqdm


def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def main(vocab_path, model_path, output_path, input_path, batch_size):
    classifier = Classifier(vocab_path=vocab_path, model_path=model_path, device=0, rs_max_len=200,
                            max_len=512)
    id_list = []
    text_list = []
    with open(output_path, 'w', encoding='utf8') as fw:
        with open(input_path, 'r', encoding="utf8") as fr:
            lines = fr.readlines()
            for step, line in enumerate(tqdm(lines)):
                data = json.loads(line)
                id_list.append(data.get('id'))
                text_list.append(data.get("text"))
                if len(id_list) == batch_size:
                    summary_list = classifier.predict(text_list)
                    for i in range(len(id_list)):
                        rst = dict(
                            id=id_list[i],
                            summary=summary_list[i]
                        )
                        fw.write(json.dumps(rst, ensure_ascii=False) + '\n')
                    id_list = []
                    text_list = []
            if len(id_list)!=0:
                summary_list = classifier.predict(text_list)
                for i in range(len(id_list)):
                    rst = dict(
                        id=id_list[i],
                        summary=summary_list[i]
                    )
                    fw.write(json.dumps(rst, ensure_ascii=False) + '\n')
                id_list = []
                text_list = []


if __name__ == "__main__":
    # 获取设备信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed=42)
    input_path = './data_dir/test.jsonl'  # origin_input file path
    vocab_path = './t5_pegasus_torch/vocab.txt'
    batch_size = 8
    if not os.path.exists("./output"):
        os.makedirs("./output")
    for i in range(10):
        print("正在测试" + str(i))
        output_path = './output/result-' + str(i) + '.jsonl'  # output file path
        model_path = './output_dir/model-' + str(i)
        main(vocab_path, model_path, output_path, input_path, batch_size)
