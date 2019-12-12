import argparse
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
# from pytorch_transformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset
import logging
from read_data import *
# from model import ClassificationXLNet
from utils import ALL_MODELS, ID2CLASS, MODEL_CLASSES
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='AAAI CLF')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--batch_size_u', default=24, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--max_seq_length', default=64, type=int, metavar='N',
                    help='max sequence length')                 

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--output_dir', default="test_model", type=str,
                    help='path to trained model and eval and test results')
parser.add_argument("--model_type", default=None, type=str, required=True,
                    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                    help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
parser.add_argument('--data-path', type=str, default='./processed_data/',
                    help='path to data folders')
parser.add_argument("--do_lower_case", action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")

args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)

best_f1 = 0

def print_score(output_scores, n_labels):
    for i in range(n_labels):
        logger.info("============================")
        logger.info("class {}".format(ID2CLASS[i]))
        result = output_scores[i]
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))


def main():
    global best_f1

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    train_labeled_set, val_set, test_set, n_labels = get_data(args.data_path, args.max_seq_length, tokenizer)
    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    #unlabeled_trainloader = Data.DataLoader(
    #    dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=512, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    config = config_class.from_pretrained(args.model_name_or_path, num_labels=n_labels)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    # model = ClassificationXLNet(n_labels).cuda()
    model.to(device)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    # optimizer = AdamW(
    # [
    #     {"params": model.module.xlnet.parameters(), "lr": args.lrmain},
    #     {"params": model.module.linear.parameters(), "lr": args.lrlast},
    # ])
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lrmain, eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
    #                                             num_training_steps=len(train_labeled_set))

    train_criterion = SemiLoss()

    all_test_f1 = []
    test_f1 = None

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(labeled_trainloader))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Model = %s" % str(args.model_name_or_path))
    logger.info("  Lower case = %s" % str(args.do_lower_case))
    logger.info("  Batch size = %d" % args.batch_size)
    logger.info("  Max seq length = %d" % args.max_seq_length)

    for epoch in trange(args.epochs, ncols=50, desc="Epoch:"):
        train(labeled_trainloader, model, optimizer, train_criterion, epoch, n_labels)

        train_output_scores, train_f1 = validate(labeled_trainloader,
                                                 model, n_labels, mode='Train Stats')

        logger.info("******Epoch {}, train score******".format(epoch))
        print_score(train_output_scores, n_labels)
        # print("epoch {}, train f1 {}".format(epoch, train_f1))

        val_output_scores, val_f1 = validate(val_loader,
                                             model, n_labels, mode='Valid Stats')

        logger.info("******Epoch {}, validation score******".format(epoch))
        print_score(val_output_scores, n_labels)
        # print("epoch {}, val f1 {}".format(epoch, val_f1))

        if sum(val_f1)/n_labels >= sum(best_f1)/n_labels:
            best_f1 = val_f1
            test_output_scores, test_f1 = validate(test_loader, model, n_labels, mode = 'Test')
            all_test_f1.append(test_f1)

            # writing results
            logger.info("******Epoch {}, test score******".format(epoch))
            output_eval_file = os.path.join(args.output_dir, "results.txt")
            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results {} *****")
                writer.write("model = %s\n" % str(args.model_name_or_path))
                writer.write(
                    "total batch size=%d\n" % args.batch_size)
                writer.write("train num epochs = %d\n" % args.epochs)
                writer.write("max seq length = %d\n" % args.max_seq_length)
                for i in range(n_labels):
                    logger.info("****************************")
                    logger.info("class {}".format(ID2CLASS[i]))
                    result = test_output_scores[i]
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))

        logger.info('Best f1:')
        logger.info(best_f1)

        logger.info('Test f1:')
        logger.info(test_f1)

    logger.info('Best f1:')
    logger.info(best_f1)

    logger.info('Test f1:')
    logger.info(test_f1)

def train(labeled_trainloader, model, optimizer,criterion, epoch, n_labels = 6):
    model.train()

    for batch_idx, (inputs , targets) in enumerate(labeled_trainloader):
        # inputs, targets = inputs.cuda(),targets.cuda(non_blocking=True)
        # outputs = model(inputs)
        batch, targets = tuple(t.to(device) for t in inputs), targets.to(device)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                  # XLM don't use segment_ids
                  }
        outputs = model(**inputs)
        outputs = outputs[0]
        # print(len(outputs))
        # print(outputs)
        loss = criterion(outputs, targets, epoch)

        optimizer.zero_grad()
        if batch_idx % 50 == 1:
            print('\nepoch {}, step {}, loss {}'.format(
                epoch, batch_idx, loss.item()))
        loss.backward()
        optimizer.step()

def validate(val_loader, model, n_labels, mode):
    model.eval()
    predict_dict = {}
    correct_dict = {}
    correct_total = {}

    for i in range(n_labels):
        predict_dict[i] = [0,0]
        correct_dict[i] = [0,0]
        correct_total[i] = [0,0]

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # outputs = model(inputs)
            batch, targets = tuple(t.to(device) for t in inputs), targets.to(device)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM don't use segment_ids
                      }
            outputs = model(**inputs)
            outputs = torch.sigmoid(outputs[0])

            id_1, id_0 = torch.where(outputs>0.5), torch.where(outputs<0.5)
            batch_size = outputs.shape[0]
            outputs[id_1] = 1
            outputs[id_0] = 0
            outputs = outputs.detach().cpu().numpy()
            targets = targets.to('cpu').numpy()
            for b in range(batch_size):
                for i in range(n_labels):
                    predict_dict[i][int(outputs[b, i])] += 1
                    correct_dict[i][int(targets[b, i])] += 1
                    if outputs[b, i] == targets[b, i]:
                        correct_total[i][int(outputs[b, i])] += 1

    all_precision = []
    all_recall = []
    all_f1 = []
    n_class = 2
    averaging = "macro"  # "pos_label"
    logger.info("averaging method: {}".format(averaging))

    for i in range(n_labels):
        precision, recall, f1 = [], [], []
        for j in range(n_class):
            p = correct_total[i][j] / predict_dict[i][j]
            r = correct_total[i][j] / correct_dict[i][j]
            f = 2 * p * r / (p + r)
            precision.append(p)
            recall.append(r)
            f1.append(f)

        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

    output_scores = []
    output_f1 = []
    for i in range(n_labels):
        if averaging == "pos_label":
            p, r, f = all_precision[i][1], all_recall[i][1], all_f1[i][1]
        elif averaging == "macro":
            p, r, f = sum(all_precision[i])/n_class, \
                      sum(all_recall[i])/n_class, sum(all_f1[i])/n_class
        else:
            raise ValueError("UnsupportedOperationException")
        output_scores.append({"precision":p, "recall":r, "f1":f})
        output_f1.append(f)

    return output_scores, output_f1


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, epoch):
        Lx = - torch.mean(torch.sum(F.logsigmoid(outputs_x) * targets_x, dim=1))
        return Lx

if __name__ == "__main__":
    main()
