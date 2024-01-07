CUDA_VISIBLE_DEVICES=7 PYTHONIOENCODING=UTF-8 python3 train_joint.py \
--num_train_epochs=10 \
--train_batch_size=8 \
--test_batch_size=8 \
--device=0 \
--pretrained_model_path=./t5_pegasus_torch/ \
--data_dir=./data_dir/ \
--train_file_path=./data_dir/train_stage2_0.5.jsonl \
--output_dir=./output_dir > train_0.5.out &

