import torch, os, logging, csv, copy, math, json

def _read_tsv(input_file, quoting=csv.QUOTE_MINIMAL):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=quoting)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def get_train_examples(data_dir):
    """See base class."""
    return _read_tsv(os.path.join(data_dir, "train.tsv"))


def get_dev_examples(data_dir):
    """See base class."""
    return _read_tsv(os.path.join(data_dir, "dev.tsv"))

def get_test_examples(data_dir):
    """See base class."""
    return _read_tsv(os.path.join(data_dir, "test.tsv"))

input_json = "../data/cnndm_entity_train.json"
output_json = "../data/cnndm_entity_rst_train.json"
with open(input_json, 'r') as f:
    data = json.load(f)


all_input_sents = [d['input'] for d in data]
all_context_sents = [d['context'] for d in data]
all_sent_labels = [d['sentlabel'] for d in data]

import os
import torch
import numpy as np
import argparse
import json
from config import *
from transformers import AutoTokenizer, AutoModel
from model_depth import ParsingNet

def inference(model, tokenizer, input_sentences, batch_size):
    LoopNeeded = int(np.ceil(len(input_sentences) / batch_size))

    input_sentences = [tokenizer.tokenize(i, add_special_tokens=False) for i in input_sentences]
    all_segmentation_pred = []
    all_tree_parsing_pred = []

    with torch.no_grad():
        for loop in range(LoopNeeded):
            StartPosition = loop * batch_size
            EndPosition = (loop + 1) * batch_size
            if EndPosition > len(input_sentences):
                EndPosition = len(input_sentences)

            input_sen_batch = input_sentences[StartPosition:EndPosition]
            
            _, _, SPAN_batch, _, predict_EDU_breaks = model.TestingLoss(input_sen_batch, input_EDU_breaks=None, LabelIndex=None,
                                                                        ParsingIndex=None, GenerateTree=True, use_pred_segmentation=True)
            all_segmentation_pred.extend(predict_EDU_breaks)
            all_tree_parsing_pred.extend(SPAN_batch)
    return input_sentences, all_segmentation_pred, all_tree_parsing_pred

bert_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
bert_model = AutoModel.from_pretrained("xlm-roberta-base")

bert_model = bert_model.cuda()

for name, param in bert_model.named_parameters():
    param.requires_grad = False

model = ParsingNet(bert_model, bert_tokenizer=bert_tokenizer)

model = model.cuda()
model.load_state_dict(torch.load('depth_mode/Savings/multi_all_checkpoint.torchsave'))
model = model.eval()


filtered_input_sents, filtered_context_sents = [], []
for input_sent, context_sent in zip(all_input_sents, all_context_sents):
    if len(bert_tokenizer.tokenize(input_sent, add_special_tokens=False)) > 2 and len(bert_tokenizer.tokenize(context_sent, add_special_tokens=False)) > 2:
        filtered_input_sents.append(input_sent)
        filtered_context_sents.append(context_sent)

context_sentences, context_segmentation_preds, context_tree_parsing_preds = inference(model, bert_tokenizer, filtered_context_sents, 1)

input_sentences, input_segmentation_preds, input_tree_parsing_preds = inference(model, bert_tokenizer, filtered_input_sents, 1)


all_data = []
for input, context, sentlabel, input_tokenized, input_segmentation_pred, input_tree_parsing_pred, context_tokenized, context_segmentation_pred, context_tree_parsing_pred in zip(all_input_sents, all_context_sents, all_sent_labels, input_sentences, input_segmentation_preds, input_tree_parsing_preds, context_sentences, context_segmentation_preds, context_tree_parsing_preds):
    data = {}
    data['input'] = input
    data['context'] = context
    data['sentlabel'] = sentlabel
    data['input_tokenized'] = input_tokenized
    data['input_segmentation_pred'] = input_segmentation_pred
    data['input_tree_parsing_pred'] = input_tree_parsing_pred
    data['context_tokenized'] = context_tokenized
    data['context_segmentation_pred'] = context_segmentation_pred
    data['context_tree_parsing_pred'] =  context_tree_parsing_pred
    all_data.append(data)

with open(output_json,  'w') as f:
    json.dump(all_data, f)