#Prepare the training data
python3 data_preprocess.py
#Model Training
CUDA_VISIBLE_DEVICES=0 nohup python3 run.py > run.out &
