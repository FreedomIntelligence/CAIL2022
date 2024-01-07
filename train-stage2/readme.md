1. 将第一阶段的数据放入data_dir文件夹中
2. 执行train_joint.py完成训练
3. 执行main.py完成测试，并使用evaluate挑选出最优模型

项目架构
    
    --data_dir 数据存放
    --DS 数据读取
    --module 模型
    --evaluate 线下评估
    --main 线下测试
    --train_joint 训练