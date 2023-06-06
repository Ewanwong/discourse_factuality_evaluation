import json
from factuality.RAEModel import load_and_cache_examples
from transformers import AutoConfig, LongformerTokenizer, AdamW
import csv
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
import pickle
import torch

with open('xsum_generation_rst_parsed_dev.json') as f:
    data = json.load(f)
print(len(data))

    
