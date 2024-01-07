# CAIL2022

## ⚡ Introduction

## ⚒️ Training
### Install the dependencies
```
 pip install -r requirements.txt
```

### Download the pretrained models
Download the pretrained XLNET and T5 models
**XLNET:**(https://huggingface.co/hfl/chinese-xlnet-mid)
**T5:**(https://huggingface.co/imxly/t5-pegasus)

### First Stage
You can train the first-stage model by:

```
 cd train-stage1
 bash run.sh
```
After training, you can prepare the data for the second stage:

1️⃣  Since we want to extract the domian-related sentences, we observe the result in the first-stage trainig with the highest F1 value for the important label (1) in the test data of each fold as the best model to be filled into `prepare_for_generate.py > best_model_index_list`.

2️⃣ Run the command to genearate `train.jsonl`:

```
CUDA_VISIBLE_DEVICES=0 nohup python3 prepare_for_generate.py > generate_data.out &
```

3️⃣ Select the high-quality sentences based your own threshold:

```
python3 select_text.py
```

### Second Stage

1️⃣ Put the data selected in the first stage (e.g. `train_stage2_0.5.jsonl`) and evaluation data in the folder `train-stage2/data_dir/`

2️⃣ Train the generative model:

```
cd ../train-stage2
bash run.sh
```
3️⃣ Select the best model on evaluation set (choose the model that performs the best during training if there is no evaluation set):

```
CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=UTF-8 nohup python3 main.py > main.out &
python3 evaluate.py > evaluate.out
```

### Evaluation
1️⃣ Put the extractive models in `e2e/extractor_model/` folder, abstractive model in `e2e/generator_model` folder and test data in the folder `e2e/data_dir/`

2️⃣ Fill the same `best_model_index_list` in `extractor.py` and their corresponsing index as **Training First Stage**

3️⃣Generate the two-stage summary:

```
cd ../e2e
bash run.sh
```

## 🏆 Awards

## 📕 Citation
