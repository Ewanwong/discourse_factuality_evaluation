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
        self.arc_embedding = nn.Embedding(42, config.hidden_size)
        self.dep_label_classifier = nn.Linear(3 * config.hidden_size, 2)


        # self.init_weights()

    def get_embeddings(self, input_ids, input_attention_mask, token_ids, child_start_indices, child_end_indices, head_start_indices, head_end_indices, arcs):

        batch_size = input_ids.shape[0]

        transformer_outputs = self.bert(input_ids, attention_mask=input_attention_mask, token_type_ids=token_ids)

        outputs = transformer_outputs[0]
        outputs = self.dropout(outputs)
        outputs = outputs.view((batch_size, -1, outputs.size(-1))) # batch, len, emb_dim
        
        child_embeddings = torch.stack([torch.mean(outputs[i, child_start_indices[i].item():child_end_indices[i].item(),:], dim=0) for i in range(batch_size)])
        head_embeddings = torch.stack([torch.mean(outputs[i, head_start_indices[i].item():head_end_indices[i].item():], dim=0) for i in range(batch_size)]) # bsz * emb_dim

        
        
        
        # arcs = arcs.view(-1, arcs.size(-1)) # bsz * len
        
        arc_outputs = self.arc_embedding(arcs)  # bsz * emb_dim
        arc_outputs = self.dropout(arc_outputs)
        
        final_embeddings = torch.cat([child_embeddings, head_embeddings, arc_outputs], dim=-1)

        return final_embeddings

    def forward(self, input_ids, input_attention_mask, token_ids, child_start_indices, child_end_indices, head_start_indices, head_end_indices, arcs):
        
        
        final_embeddings = self.get_embeddings(input_ids, input_attention_mask, token_ids, child_start_indices, child_end_indices, head_start_indices, head_end_indices, arcs) # n * dim

        logits_all = self.dep_label_classifier(final_embeddings) # n  * 2

        return logits_all
    



