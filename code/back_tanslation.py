import torch
import fairseq
import pickle
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
print(de2en.translate(en2de.translate('Hello world!')))

data_path = './processed_data/'

with open(data_path + 'train_unlabeled_data.pkl', 'rb') as f:
    train_unlabeled_data = pickle.load(f)

num_sample_sen = 3
for key, value in train_unlabeled_data.items():
    for i in range(num_sample_sen):
        new_value = de2en.translate(en2de.translate(value[1]))
        value.append(new_value)

with open(data_path + 'train_unlabeled_data_bt.pkl', 'wb') as f:
    assert len(train_unlabeled_data[0]) == num_sample_sen + 2
    pickle.dump(train_unlabeled_data, f)

