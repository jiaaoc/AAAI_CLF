import torch
import torch.nn as nn
from pytorch_transformers import *


class ClassificationXLNet(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(ClassificationXLNet, self).__init__()

        self.xlnet = XLNetModel.from_pretrained(model_name)
        # self.transformer = transformer_model
        self.max_pool = nn.MaxPool1d(64)

        self.linear = nn.Linear(768, num_labels)

    def forward(self, x):
        # print("x: ", x.shape)
        all_hidden, pooler = self.xlnet(x)
        # outputs = self.transformer(**x)
        # all_hidden = outputs[0]
        # print("allh: ", all_hidden.shape)

        pooled_output = self.max_pool(all_hidden.transpose(1, 2))
        pooled_output = pooled_output.squeeze(2)

        predict = self.linear(pooled_output)
        return predict