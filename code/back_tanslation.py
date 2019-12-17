import torch
import fairseq
import pickle
from tqdm import tqdm
import os
# List available models
print(torch.hub.list('pytorch/fairseq'))  # [..., 'transformer.wmt16.en-de', ... ]

# Load a transformer trained on WMT'16 En-De
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',  checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en',  checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe')
# en2de = TransformerModel.from_pretrained(
#   '/nethome/ywu825/AAAI_CLF/trans_models',
#   checkpoint_file='model1.pt',
#   data_name_or_path='data-bin/wmt17_zh_en_full',
#   bpe='subword_nmt',
#   bpe_codes='bpecodes'
# )

# The underlying model is available under the *models* attribute
# print(type(en2de))
# print("------------------------------------------")
# raise ValueError()
assert isinstance(en2de.models[0], fairseq.models.transformer.TransformerModel)

# Translate a sentence
# print(de2en.translate(en2de.translate('I am yuwei. How are you?', sampling = True, temperature = 0.7),
#                                       sampling = True, temperature = 0.7))

data_path = './processed_data/'

with open(data_path + 'train_unlabeled_data.pkl', 'rb') as f:
    train_unlabeled_data = pickle.load(f)

num_sample_sen = 2
cnt = 0
train_unlabeled_data_aug = {}
gpu = "5"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
use_cuda = torch.cuda.is_available()
en2de = en2de.cuda()
de2en = de2en.cuda()

for key, value in tqdm(train_unlabeled_data.items(), ncols=50, desc="Iteration:"):
    new_value = []
    if cnt <= 69000:
        cnt += 1
        continue
    for i in range(num_sample_sen):
        v = de2en.translate(en2de.translate(value[1], sampling = True, temperature = 0.8),
                                    sampling = True, temperature = 0.8)
        if cnt % 100 == 0:
            print("***************")
            print("org: ", value[1])
            print("new: ", v)
        new_value.append(v)
    train_unlabeled_data_aug[key] = new_value
    if cnt % 1000 == 0:
        with open(data_path + 'train_unlabeled_data_bt_{}.pkl'.format(cnt), 'wb') as f:
            assert len(train_unlabeled_data_aug[key]) == num_sample_sen
            pickle.dump(train_unlabeled_data_aug, f)
    cnt += 1

with open(data_path + 'train_unlabeled_data_bt_{}.pkl'.format(cnt), 'wb') as f:
    pickle.dump(train_unlabeled_data_aug, f)


