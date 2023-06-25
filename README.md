在选取数据集的时候，在utils.py中的get_df()函数改变数据集的路径调整是否加入prompt,在lstm_train.py中通过改变values的值选择什么时候加入MLP的辅助分类信息
使用jieba分词，去除停用词；
预训练词向量"chinese-word-vectors"，github链接为：https://github.com/Embedding/Chinese-Word-Vectors; 下载链接为https://pan.baidu.com/s/11PWBcvruXEDvKf2TiIXntg；
使用的是python3.8.16
使用pip install -r requirements.txt安装包
运行python lstm_train.py运行LSTM的训练文件
运行python bert.py运行bert的训练文件
