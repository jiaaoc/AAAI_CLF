from transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer,
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,
                                  )

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, XLNetConfig, AlbertConfig)), ())
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer)}

ID2CLASS = {0:"Emotional_disclosure", 1:"Information_disclosure",
            2:"Support", 3:"General_support", 4:"Info_support", 5:"Emo_support"}
