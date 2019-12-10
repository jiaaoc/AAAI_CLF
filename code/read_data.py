import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_transformers import *
import torch.utils.data as Data

def get_data(data_path, max_seq_len, model = 'xlnet-base-cased'):
    tokenizer = XLNetTokenizer.from_pretrained(model)

    with open(data_path + 'labeled_data.pkl', 'rb') as f:
        labeled_data = pickle.load(f)
    
    with open(data_path + 'test_unlabeled_data.pkl', 'rb') as f:
        test_unlabeled_data = pickle.load(f)

    with open(data_path + 'train_unlabeled_data.pkl', 'rb') as f:
        train_unlabeled_data = pickle.load(f)

    n_class = 6
    
    labeled_ids = list(labeled_data.keys())
    np.random.seed(0)

    np.random.shuffle(labeled_ids)
    labeled_train_ids = labeled_ids[:-4860]
    labeled_dev_ids = labeled_ids[-4860:-2860]
    labeled_test_ids = labeled_ids[-2860:]

    train_labeled_dataset = loader_labeled(
        labeled_data, labeled_train_ids, tokenizer, max_seq_len)
    val_dataset = loader_labeled(
        labeled_data, labeled_dev_ids, tokenizer, max_seq_len)
    test_dataset = loader_labeled(
        labeled_data, labeled_test_ids, tokenizer, max_seq_len)

    print("#Labeled: {}, Unlabeled {}, Val {}, Test {}".format(len(labeled_train_ids), len(train_unlabeled_data), len(labeled_dev_ids), len(labeled_test_ids)))

    return train_labeled_dataset, val_dataset, test_dataset, n_class



class loader_labeled(Dataset):
    def __init__(self, data, ids, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.data = data
        self.ids = ids
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.ids)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result
    
    def __getitem__(self, idx):
        sent_id = self.ids[idx]
        text = self.data[sent_id][1]
        l = self.data[sent_id][2]
        encode_result = self.get_tokenized(text)

        labels = [0,0,0,0,0,0]

        for i in range(0, len(l)):
            labels[l[i]] = 1
        
        return (torch.tensor(encode_result), labels)







    