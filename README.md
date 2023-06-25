# 这是笔者在电子科技大学本科课程人工智能实验三（2023summer）期间的课程代码的任务二部分

在选取数据集的时候，在utils.py中的get_df()函数改变数据集的路径调整是否加入prompt,在lstm_train.py中通过改变values的值选择什么时候加入MLP的辅助分类信息

使用jieba分词，去除停用词；

预训练词向量"chinese-word-vectors"，

github链接为：https://github.com/Embedding/Chinese-Word-Vectors; 

下载本次实验的链接为https://pan.baidu.com/s/11PWBcvruXEDvKf2TiIXntg；

创建一个名为embeddings的文件夹，然后把下载的权重放到./embeddings文件夹中

使用的是python3.8.16

使用pip install -r requirements.txt安装依赖环境

运行python lstm_train.py运行LSTM的训练文件

运行python bert.py运行bert的训练文件

训练结束后可在新自动创建的./runs文件夹下查看相关训练结果和日志文件
