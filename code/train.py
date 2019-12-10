import argparse
import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from pytorch_transformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset

from read_data import *
from model import ClassificationXLNet


parser = argparse.ArgumentParser(description='AAAI CLF')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--batch-size-u', default=24, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--max-seq-len', default=64, type=int, metavar='N',
                    help='max sequence length')                 

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')


parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--model', type=str, default='xlnet-base-cased',
                    help='pretrained model')   

parser.add_argument('--data-path', type=str, default='./processed_data/',
                    help='path to data folders')                   

args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)

best_f1 = 0

def main():
    global best_f1
    train_labeled_set, val_set, test_set, n_labels = get_data(args.data_path, args.max_seq_len, model = args.model )
    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    #unlabeled_trainloader = Data.DataLoader(
    #    dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=512, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=512, shuffle=False)


    model = ClassificationXLNet(n_labels).cuda()
    model = nn.DataParallel(model)

    optimizer = AdamW(
    [
        {"params": model.module.xlnet.parameters(), "lr": args.lrmain},
        {"params": model.module.linear.parameters(), "lr": args.lrlast},
    ])
    
    train_criterion = SemiLoss()

    test_f1 = []

    for epoch in range(args.epochs):
        train(labeled_trainloader, model, optimizer, train_criterion, epoch, n_labels)


        train_f1 = validate(labeled_trainloader,
                                model, epoch, n_labels, mode='Train Stats')
        
        print("epoch {}, train f1 {}".format(epoch, train_f1))

        val_f1 = validate(val_loader, model, epoch,n_labels, mode='Valid Stats')

        print("epoch {}, val f1 {}".format(epoch, val_f1))

        if val_f1 >= best_f1:
            best_f1 = val_f1
            test_f1_ = validate(test_loader, model, epoch, n_labels, mode = 'Test')
            test_f1.append(test_f1_)
            print("epoch {}, test f1 {}".format(epoch, test_f1_))

        print('Best f1:')
        print(best_f1)

        print('Test f1:')
        print(test_f1)

    print('Best f1:')
    print(best_f1)

    print('Test f1:')
    print(test_f1)

def train(labeled_trainloader, model, optimizer,criterion, epoch, n_labels = 6):
    model.train()

    for batch_idx, (inputs, targets) in enumerate(labeled_trainloader):
        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        print('epoch {}, step {}, loss {}'.format(
            epoch, batch_idx, loss.item()))
        loss.backward()
        optimizer.step()

def validate(val_loader, model, epoch, n_labels,mode):
    model.eval()

    predict_dict = {}
    correct_dict = {}
    correct_total = {}

    for i in range(0, n_labels):
        predict_dict[i] = [0,0]
        correct_dict[i] = [0,0]
        correct_total[i] = [0,0]
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            
            outputs = F.sigmoid(outputs)
            
            id_1, id_0 = torch.where(a > 0.5), torch.where(a<0.5)
            outputs[id_1] = 1
            outputs[id_0] = 0

            for i in range(0, 6):
                predict_dict[i][outputs[:, i]] += 1
                correct_dict[i][targets[:, i]] += 1
                if outputs[:, i] == targets[:, i]:
                    correct_total[i][outputs[:, i]] += 1

    f1 = []  
    for i in range(0, n_labels):
        



            



            



class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, epoch):
        Lx = - torch.mean(torch.sum(F.logsigmoid(outputs_x) * targets_x, dim=1))
        return Lx