import torch
import utils as utils
from torch.utils.data import  Dataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import config as config
class RummorDataset(Dataset):

    def __init__(self,model="train"):
        super(RummorDataset, self).__init__()

        self.contents,self.labels,self.comments,self.likes,self.reposts =utils.get_df()
        self.comments,self.likes,self.reposts=torch.tensor(self.comments).reshape(1, -1),torch.tensor(self.likes).reshape(1, -1),torch.tensor(self.reposts).reshape(1, -1)
        minmax_c = MinMaxScaler()
        minmax_c.fit(self.comments)
        self.comments = minmax_c.transform(self.comments)
        minmax_l = MinMaxScaler()
        minmax_l.fit(self.likes)
        self.likes = minmax_l.transform(self.likes)
        minmax_r = MinMaxScaler()
        minmax_r.fit(self.reposts)
        self.reposts = minmax_r.transform(self.reposts)
        self.contents=utils.key_to_index(self.contents,utils.word2vec,config.num_words)

        self.maxlen=utils.get_maxlength(self.contents)

        self.contents=utils.padding_truncating(self.contents,self.maxlen)
        minmax_c = MinMaxScaler()
        x_train,x_test,c_train,c_test,l_train,l_test,r_train,r_test,y_train,y_test=train_test_split(self.contents,self.comments.reshape(-1).tolist(),self.likes.reshape(-1).tolist(),self.reposts.reshape(-1).tolist(),self.labels,test_size=0.2,shuffle=True,random_state=0)
        if model=="train":
            self.contents=x_train
            self.comments=c_train
            self.likes=l_train
            self.reposts=r_train
            self.labels=y_train
        elif model=="test":
            self.contents = x_test
            self.comments =c_test
            self.likes=l_test
            self.reposts=r_test
            self.labels = y_test

    def __getitem__(self, item):
        return torch.tensor(self.contents[item]),torch.tensor(self.comments[item]).view(-1,1),torch.tensor(self.likes[item]).view(-1,1),torch.tensor(self.reposts[item]).view(-1,1),torch.tensor(self.labels[item])

    def __len__(self):
        return len(self.contents)

def get_dataloader(model="train"):
    dataset=RummorDataset(model=model)
    return DataLoader(dataset,batch_size=config.batch_size,shuffle=True if model=="train" else False)


