from rouge.rouge import Rouge
from tqdm import tqdm
import json
import sys
sys.setrecursionlimit(10000000)

class Selector:
    def __init__(self):
        self.rouge = Rouge()
        self.result_select = []
        self.answer = {"golden": {}, "predict": {}}
        
        

    def read_answer(self, file_name, mode="golden"):
        if mode == "golden":
            with open(file_name, 'r', encoding="utf8") as f:
                for line in f:
                    data = json.loads(line)
                    self.answer[mode][data.get("id")] = data.get('summary')
        else:
            with open(file_name, 'r', encoding="utf8") as f:
                for line in f:
                    data = json.loads(line)
                    self.answer[mode][data.get("id")] = data.get('text')

    def compute_rouge(self, source, target):
        """计算rouge-1、rouge-2、rouge-l
        """
        #print(source, target)
        source, target = ' '.join(source), ' '.join(target)
        
        try:
            scores = self.rouge.get_scores(hyps=source, refs=target)
            return {
                'rouge-1': scores[0]['rouge-1']['f'],
                'rouge-2': scores[0]['rouge-2']['f'],
                'rouge-l': scores[0]['rouge-l']['f'],
            }
        except ValueError:
            return {
                'rouge-1': 0.0,
                'rouge-2': 0.0,
                'rouge-l': 0.0,
            }

    def compute_rouges(self,threshold):
        sources = self.answer["golden"]
        targets = self.answer["predict"]
        for id, source in sources.items():
            scores = {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }
            target = targets[id]
            score = self.compute_rouge(source, target)
            for k, v in scores.items():
                scores[k] = v + score[k]
            rouge_all= 0.2 * scores["rouge-1"] + 0.3 * scores["rouge-2"] + 0.5 * scores["rouge-l"]
            if rouge_all >= threshold:
                self.result_select.append({"id":id,"text":target,"summary":source})
                #print("Good sample")
            else:
                pass
                #print("Bad sample!")
        
    def write_select(self,result_path):
        for sample in self.result_select:  
            with open(result_path,"a",encoding="utf-8") as output:
                output.write(json.dumps(sample,ensure_ascii=False))
                output.write("\n")


if __name__ == '__main__':
    input_path = './origin_data/train.jsonl'  # origin_input file path
    output_path = 'train.jsonl'  # output file path
    threshold = 0.5
    result_path = "train_stage2_" + str(threshold) + ".jsonl"
    selector = Selector()
    selector.read_answer(input_path, "golden")
    selector.read_answer(output_path, "predict")
    selector.compute_rouges(threshold)
    selector.write_select(result_path)
    print("finished")
