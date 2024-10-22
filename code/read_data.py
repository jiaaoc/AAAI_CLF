import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_transformers import *
import torch.utils.data as Data
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def get_data(data_path, max_seq_len, tokenizer, no_class, if_pred=False):
    # tokenizer = XLNetTokenizer.from_pretrained(model)

    with open(data_path + 'labeled_data.pkl', 'rb') as f:
        labeled_data = pickle.load(f)
    
    with open(data_path + 'test_unlabeled_data.pkl', 'rb') as f:
        test_unlabeled_data = pickle.load(f)

    with open(data_path + 'train_unlabeled_data.pkl', 'rb') as f:
        train_unlabeled_data = pickle.load(f)

#     with open(data_path + 'train_unlabeled_data_bt_69000.pkl', 'rb') as f:
#         train_unlabeled_aug_data = pickle.load(f)

    n_class = 6
    
    labeled_ids = list(labeled_data.keys())
    np.random.seed(0)

    np.random.shuffle(labeled_ids)
    labeled_train_ids = labeled_ids[:-4860]
    labeled_dev_ids = labeled_ids[-4860:-2860]
    labeled_test_ids = labeled_ids[-2860:]
    if if_pred:
        labeled_train_ids += labeled_test_ids
    unlabeled_ids = list(train_unlabeled_data.keys())
    unlabeled_train_ids = unlabeled_ids[:69001]
    analyzer = SentimentIntensityAnalyzer()

    test_unlabeled_ids = list(test_unlabeled_data.keys())

    train_labeled_dataset = loader_labeled(
        labeled_data, labeled_train_ids, tokenizer, max_seq_len, no_class, analyzer.polarity_scores)
    train_unlabeled_dataset = loader_unlabeled(
        train_unlabeled_data, unlabeled_train_ids, tokenizer, max_seq_len, analyzer.polarity_scores)
#     train_unlabeled_aug_dataset = loader_unlabeled(
#         train_unlabeled_aug_data, unlabeled_train_ids, tokenizer, max_seq_len, analyzer.polarity_scores)
    train_unlabeled_aug_dataset = None
    val_dataset = loader_labeled(
        labeled_data, labeled_dev_ids, tokenizer, max_seq_len, no_class, analyzer.polarity_scores)
    test_dataset = loader_labeled(
        labeled_data, labeled_test_ids, tokenizer, max_seq_len, no_class, analyzer.polarity_scores)
    test_unlabeled_dataset = loader_unlabeled(test_unlabeled_data, test_unlabeled_ids, tokenizer,
                                              max_seq_len, analyzer.polarity_scores)
    print("#Labeled: {}, Unlabeled {}, Val {}, Test {}".format(len(labeled_train_ids), len(train_unlabeled_data), len(labeled_dev_ids), len(labeled_test_ids)))

    if if_pred:
        return train_labeled_dataset, train_unlabeled_dataset, train_unlabeled_aug_dataset, val_dataset, \
               test_unlabeled_dataset, n_class

    return train_labeled_dataset, train_unlabeled_dataset, train_unlabeled_aug_dataset, val_dataset, test_dataset, n_class


def get_text_data(data_path, max_seq_len, tokenizer, no_class):

    with open(data_path + 'labeled_data.pkl', 'rb') as f:
        labeled_data = pickle.load(f)
    labeled_ids = list(labeled_data.keys())
    np.random.seed(0)

    np.random.shuffle(labeled_ids)
    labeled_train_ids = labeled_ids[:-4860]

    train_labeled_dataset = loader_text_labeled(
        labeled_data, labeled_train_ids, tokenizer, max_seq_len, no_class)


    return train_labeled_dataset



class loader_labeled(Dataset):
    def __init__(self, data, ids, tokenizer, max_seq_len, no_class, analyzer):
        self.tokenizer = tokenizer
        self.data = data
        self.ids = ids
        self.max_seq_len = max_seq_len
        self.no_class = no_class
        self.analyzer = analyzer

    def __len__(self):
        return len(self.ids)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        sen_dict = self.analyzer(text)
        sen_score = [sen_dict['pos'], sen_dict['neg'], sen_dict['neu'], sen_dict['compound']]
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        mask = [1] * len(encode_result)
        segement_id = [1] * len(encode_result)
        padding = [0] * (self.max_seq_len - len(encode_result))

        encode_result += padding
        mask += padding
        segement_id += padding

        return encode_result, mask, segement_id, sen_score
    
    def __getitem__(self, idx):
        sent_id = self.ids[idx]
        text = self.data[sent_id][1]
        l = self.data[sent_id][2]
        encode_result, mask, segement_id, sen_score = self.get_tokenized(text)

        labels = [0,0,0,0,0,0]

        for i in range(0, len(l)):
            labels[l[i]] = 1
        labels = labels[self.no_class]
        # return (torch.tensor(encode_result), torch.tensor(mask), \
        #        torch.tensor(segement_id)), torch.tensor(labels)
        return torch.tensor(encode_result), torch.tensor(labels), torch.tensor(sen_score)


class loader_text_labeled(Dataset):
    def __init__(self, data, ids, tokenizer, max_seq_len, no_class):
        self.tokenizer = tokenizer
        self.data = data
        self.ids = ids
        self.max_seq_len = max_seq_len
        self.no_class = no_class

    def __len__(self):
        return len(self.ids)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        mask = [1] * len(encode_result)
        segement_id = [1] * len(encode_result)
        padding = [0] * (self.max_seq_len - len(encode_result))

        encode_result += padding
        mask += padding
        segement_id += padding

        return encode_result, mask, segement_id

    def __getitem__(self, idx):
        sent_id = self.ids[idx]
        text = self.data[sent_id][1]
        l = self.data[sent_id][2]
        # encode_result, mask, segement_id = self.get_tokenized(text)

        labels = [0, 0, 0, 0, 0, 0]

        for i in range(0, len(l)):
            labels[l[i]] = 1
        labels = labels[self.no_class]
        # return (torch.tensor(encode_result), torch.tensor(mask), \
        #        torch.tensor(segement_id)), torch.tensor(labels)
        return text, labels

class loader_unlabeled(Dataset):
    def __init__(self, data, ids, tokenizer, max_seq_len, analyzer):
        self.tokenizer = tokenizer
        self.data = data
        self.ids = ids
        self.max_seq_len = max_seq_len
        self.analyzer = analyzer

    def __len__(self):
        return len(self.ids)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        sen_dict = self.analyzer(text)
        sen_score = [sen_dict['pos'], sen_dict['neg'], sen_dict['neu'], sen_dict['compound']]
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, sen_score


    def __getitem__(self, idx):
        sent_id = self.ids[idx]
        text = self.data[sent_id][0]
        text2 = self.data[sent_id][1]
        # print(text)
        # print("2: ", text2)
        # l = self.data[sent_id][2]
        encode_result, sen_score = self.get_tokenized(text)
        encode_result2, sen_score2 = self.get_tokenized(text2)

        # labels = [0, 0, 0, 0, 0, 0]

        # for i in range(0, len(l)):
        #     labels[l[i]] = 1
        # labels = labels[0]
        return (torch.tensor(encode_result), torch.tensor(sen_score)), (torch.tensor(encode_result2), torch.tensor(sen_score2))#, torch.tensor(labels)