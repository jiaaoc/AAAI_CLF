import torch
import torch.nn as nn
from pytorch_transformers import *


class ClassificationXLNet(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(ClassificationXLNet, self).__init__()

        self.transformer = XLNetModel.from_pretrained(model_name)
        self.max_pool = nn.MaxPool1d(64)
        self.drop = nn.Dropout(0.3)
        self.linear = nn.Sequential(nn.Linear(768+4, num_labels))

    def forward(self, x, sen_x):
        # print("x: ", x.shape)
        # all_hidden, pooler = self.xlnet(x)
        all_hidden, pooler = self.transformer(x)
        pooled_output = torch.mean(all_hidden, 1)
        sen_output = torch.cat([pooled_output, sen_x], dim=-1)
        
        predict = self.linear(sen_output)
        predict = self.drop(predict)

        return predict

    
class ClassificationBERT(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(ClassificationBERT, self).__init__()
        
        self.transformer = BertModel.from_pretrained(model_name)
        self.max_pool = nn.MaxPool1d(64)
        self.drop = nn.Dropout(0.3)
        self.linear = nn.Sequential(nn.Linear(768+4, num_labels))

    def forward(self, x, sen_x):
        all_hidden, pooler = self.transformer(x)
        pooled_output = torch.mean(all_hidden, 1)
        sen_output = torch.cat([pooled_output, sen_x], dim=-1)
        
        predict = self.linear(sen_output)
        predict = self.drop(predict)

        return predict