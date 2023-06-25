import torch
from lstm_model import LSTM_Model,MLP
from dataset import RummorDataset
import config as config
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataset import  get_dataloader
from torch.nn import CrossEntropyLoss
from simpletransformers.classification import ClassificationModel
import pandas as pd
import matplotlib.pyplot as plt
import  torch
import numpy as np   #gensim用来加载预训练词向量
import config as config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
values=-1  #在第几个epoch加入MLP的辅助信息
def train(epoch,model,mlp,loss_fn,optimizer,train_dataloader):
    model.train()
    loss_list = []
    train_acc = 0
    train_total = 0
    loss_fn.to(device)
    bar = tqdm(train_dataloader, total=len(train_dataloader))  #配置进度条
    for idx, (input, comment,like,repost,target) in enumerate(bar):
        input = input.to(device)
        target = target.to(device)
        if epoch > values:            
            feature=torch.hstack((comment,like,repost)).squeeze()        
            feature=feature.to(device)          
            output = model(input)
            fea=mlp(feature)
            output=output*fea
            loss =loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            loss_list.append (loss.cpu().item())
            optimizer.step()
        else:
            output = model(input)
            loss =loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            loss_list.append (loss.cpu().item())
            optimizer.step()
        # 准确率
        output_max = output.max (dim=-1)  # 返回最大值和对应的index
        pred = output_max[-1]  # 最大值的index
        train_acc += pred.eq (target).cpu ().float().sum ().item ()
        train_total += target.shape[0]
    acc = train_acc / train_total
    print("train epoch:{}  loss:{:.6f} acc:{:.5f}".format(epoch, np.mean(loss_list),acc))
    return acc,np.mean(loss_list)


def test(epoch,model,mlp,loss_fn,test_dataloader):
    model.eval()
    loss_list = []
    test_acc=0
    test_total=0
    loss_fn.to (device)
    with torch.no_grad():
        for input, comment,like,repost,target in test_dataloader:
            input = input.to(device)
            target = target.to(device)
            if epoch>values:
                feature=torch.hstack((comment,like,repost)).squeeze()        
                feature=feature.to(device)          
                fea=mlp(feature)
                output = model(input)
                output=output*fea
            else:
                output = model(input)
            loss = loss_fn(output, target)
            loss_list.append(loss.item())
            # 准确率
            output_max = output.max(dim=-1) #返回最大值和对应的index
            pred = output_max[-1]  #最大值的index
            # test_acc+=pred.eq(target).cpu().float().sum().item()
            test_acc+=pred.eq(target).cpu().sum().item()
            test_total+=target.shape[0]
        acc=test_acc/test_total
        print("test loss:{:.6f},acc:{}".format(np.mean(loss_list), acc))
    return acc,np.mean(loss_list)


if __name__ == '__main__':
    acc_all=[]
    for i in range(10):
        model=LSTM_Model()
        mlp=MLP(n_feature=3,n_hidden=16,n_output=2)
        mlp=mlp.to(device)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        train_dataloader = get_dataloader(model='train')
        test_dataloader = get_dataloader(model='test')
        loss_fn=CrossEntropyLoss()
        best_acc=0
        early_stop_cnt=0
        train_loss_list=[]
        test_loss_list=[]
        for epoch in range(config.epoch):
            train_acc,train_loss=train(epoch,model,mlp,loss_fn,optimizer,train_dataloader)
            test_acc,test_loss=test(epoch,model,mlp,loss_fn,test_dataloader)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            if test_acc>best_acc:
                best_acc=test_acc
                torch.save(model.state_dict(), 'model/model.pkl')
                print("save model,acc:{}".format(best_acc))
                early_stop_cnt=0
            else:
                early_stop_cnt+=1
            if early_stop_cnt>config.early_stop_cnt:
                break
        acc_all.append(best_acc)
    print(acc_all)
    print(sum(acc_all)/len(acc_all)) 

