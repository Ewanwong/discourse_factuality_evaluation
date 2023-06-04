from torch import nn
import torch, os, logging, csv, copy, math
from transformers.models.longformer.modeling_longformer import LongformerForMaskedLM, LongformerPreTrainedModel, LongformerModel
from transformers import LongformerTokenizer
from transformers.models.electra.modeling_electra import ElectraPreTrainedModel, ElectraModel
from torch.utils.data import TensorDataset
from torch.nn import CrossEntropyLoss
import json
import random
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
import numpy as np
logger = logging.getLogger(__name__)



class RAEBaseModel(LongformerPreTrainedModel):
    def __init__(self, config):
        super(RAEBaseModel, self).__init__(config)
        self.bert = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.arc_embedding = nn.Sequential(nn.Embedding(42, 168), nn.Tanh(), nn.Linear(168, config.hidden_size))
        self.dep_label_classifier = nn.Sequential(nn.Linear(2 * 3 * config.hidden_size, 512), nn.ReLU(), nn.Linear(512, 2))


        # self.init_weights()

    def get_embeddings(self, input_ids, input_attention_mask, token_ids, input_child_start_indices, input_child_end_indices, input_head_start_indices, input_head_end_indices, input_arcs, context_child_start_indices, context_child_end_indices, context_head_start_indices, context_head_end_indices, context_arcs):

        batch_size = input_ids.shape[0]

        transformer_outputs = self.bert(input_ids, attention_mask=input_attention_mask, token_type_ids=token_ids)

        outputs = transformer_outputs[0]
        outputs = self.dropout(outputs)
        outputs = outputs.view((batch_size, -1, outputs.size(-1))) # batch, len, emb_dim
        
        input_child_embeddings = torch.stack([torch.mean(outputs[i, input_child_start_indices[i].item():input_child_end_indices[i].item(),:], dim=0) for i in range(batch_size)])
        input_head_embeddings = torch.stack([torch.mean(outputs[i, input_head_start_indices[i].item():input_head_end_indices[i].item():], dim=0) for i in range(batch_size)]) # bsz * emb_dim

        # arcs = arcs.view(-1, arcs.size(-1)) # bsz * len
        
        input_arc_outputs = self.arc_embedding(input_arcs)  # bsz * emb_dim
        input_arc_outputs = self.dropout(input_arc_outputs)

        
        context_child_embeddings = torch.stack([torch.mean(outputs[i, context_child_start_indices[i].item():context_child_end_indices[i].item(),:], dim=0) for i in range(batch_size)])
        context_head_embeddings = torch.stack([torch.mean(outputs[i, context_head_start_indices[i].item():context_head_end_indices[i].item():], dim=0) for i in range(batch_size)]) # bsz * emb_dim

        # arcs = arcs.view(-1, arcs.size(-1)) # bsz * len
        
        context_arc_outputs = self.arc_embedding(context_arcs)  # bsz * emb_dim
        context_arc_outputs = self.dropout(context_arc_outputs)
        
        final_embeddings = torch.cat([input_child_embeddings, input_head_embeddings, input_arc_outputs, context_child_embeddings, context_head_embeddings, context_arc_outputs], dim=-1)

        return final_embeddings

    def forward(self, input_ids, input_attention_mask, token_ids, child_start_indices, child_end_indices, head_start_indices, head_end_indices, arcs):
        
        
        final_embeddings = self.get_embeddings(input_ids, input_attention_mask, token_ids, child_start_indices, child_end_indices, head_start_indices, head_end_indices, arcs) # n * dim

        logits_all = self.dep_label_classifier(final_embeddings) # n  * 2

        return logits_all