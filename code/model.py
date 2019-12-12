import torch
import torch.nn as nn
from pytorch_transformers import *


class ClassificationXLNet(nn.Module):
    def __init__(self, num_labels=2):
        super(ClassificationXLNet, self).__init__()

        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')

        self.max_pool = nn.MaxPool1d(64)

        self.linear = nn.Linear(768, num_labels)

    def forward(self, x):

        all_hidden, pooler = self.xlnet(x)
        pooled_output = self.max_pool(all_hidden.transpose(1, 2))
        pooled_output = pooled_output.squeeze(2)

        predict = self.linear(pooled_output)

        return predict