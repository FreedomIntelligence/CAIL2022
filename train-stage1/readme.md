1. 训练数据，测试数据（线下），和评估数据（线上）都放置在origin_data里了。
2. 运行data_preprocess.py完成数据的准备
3. 运行run.py完成模型的训练和测试
4. 由于我们想抽取出重要的句子（1），因此观察每折中测试数据中1的F1值最大的模型作为best model填入运行prepare_for_generate.py里
5. 运行prepare_for_generate.py完成第二阶段的数据准备

项目架构
    
    --data_dir 预处理数据存放

    --module 模型架构

    --origin_data 原始数据

    --ten_fold_data_dir 十倍交叉验证数据目录、模型存储目录
    
    --data_preprocess.py  数据准备入口
    
    --run.py  模型训练、测试入口
    
    -- prepare_for_generate.py 准备第二阶段数据入口