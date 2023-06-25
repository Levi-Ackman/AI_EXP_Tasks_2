import torch
import config as config
from simpletransformers.classification import ClassificationModel
import pandas as pd
import  torch
import  os
import numpy as np
import config as config
import random
import json
# define hyperparameter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import time
from sklearn.utils import shuffle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_df():
    weibo = pd.read_csv('./prompt-data.csv',  engine='python' ,  index_col=0)
    weibo=weibo[['text', 'rumor']]
    weibo=shuffle(weibo)
    return weibo

def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')

if __name__ == '__main__':
    results=[]
    for i in range(10):
        train_args ={"reprocess_input_data": True,
                "fp16":False,
                "num_train_epochs":20,
                "logging_steps":1,
                #"evaluate_during_training":1
                }

        # Create a ClassificationModel
        model = ClassificationModel(
            "bert", 'bert-base-chinese',
            num_labels=2,
            args=train_args
        )
        df=get_df()
        train_df,test_df=train_test_split(df,test_size=0.2,shuffle=True,random_state=0) 
        #print(train_df)
        print('train shape: ',train_df.shape)
        print('test shape: ',test_df.shape)
        begin=time.time()
        model.train_model(train_df,output_dir='output')
        end=time.time()
        print('time:{:.4f}s'.format(end-begin))
        result, model_outputs, wrong_predictions = model.eval_model(test_df, f1=f1_multiclass, acc=accuracy_score)
        print(result)
        results.append(result['acc'])
    print('results',results)
    print(np.sum(results)/10)
    

