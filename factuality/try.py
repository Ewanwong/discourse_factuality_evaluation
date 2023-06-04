import json
from RAEModel import load_and_cache_examples
from transformers import AutoConfig, LongformerTokenizer, AdamW
import torch
from torch.utils.data import DataLoader
from data_finetune import CombinedDataset
"""
path = 'cnndm_human_rst_parsed_test.json'
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
feature_path = 'cnndm_human_rst_parsed_test.pt'
dataloader = load_and_cache_examples(path, feature_path, 2048, tokenizer, shuffle=False)
l = 0
for i in dataloader:
    l += 1
print(l)
"""

train_dataset = torch.load('cnndm_train.pt')
train_dataset.remove_columns(['article'])
train_dataset.remove_columns(['highlights'])
train_dataset.set_format('torch')
train_dataloader = DataLoader(train_dataset, batch_size=2)
for batch in train_dataloader:
    print(batch['input_ids'])
    break