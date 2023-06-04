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



RelationTable = ['Attribution Satellite Nucleus', 'Enablement Nucleus Satellite', 'Cause Satellite Nucleus', 'Cause Nucleus Nucleus', 'Temporal Satellite Nucleus', 'Condition Nucleus Nucleus', 'Cause Nucleus Satellite', 'Elaboration Nucleus Satellite', 'Background Nucleus Satellite', 'Topic-Comment Satellite Nucleus', 'Elaboration Satellite Nucleus', 'Evaluation Satellite Nucleus', 'Explanation Nucleus Nucleus', 'TextualOrganization Nucleus Nucleus', 'Background Satellite Nucleus', 'Contrast Nucleus Nucleus', 'Evaluation Nucleus Satellite', 'Topic-Comment Nucleus Nucleus', 'Condition Nucleus Satellite', 'Comparison Nucleus Satellite', 'Explanation Satellite Nucleus', 'Contrast Nucleus Satellite', 'Comparison Satellite Nucleus', 'Condition Satellite Nucleus', 'Summary Satellite Nucleus', 'Explanation Nucleus Satellite', 'Enablement Satellite Nucleus', 'Temporal Nucleus Nucleus', 'Temporal Nucleus Satellite', 'Topic-Comment Nucleus Satellite', 'Manner-Means Nucleus Satellite', 'Same-Unit Nucleus Nucleus', 'Summary Nucleus Satellite', 'Contrast Satellite Nucleus', 'Attribution Nucleus Satellite', 'Manner-Means Satellite Nucleus', 'Joint Nucleus Nucleus', 'Comparison Nucleus Nucleus', 'Evaluation Nucleus Nucleus', 'Topic-Change Nucleus Nucleus', 'Topic-Change Nucleus Satellite', 'Summary Nucleus Nucleus']




class RAEBaseModel(LongformerPreTrainedModel):
    def __init__(self, config):
        super(RAEBaseModel, self).__init__(config)
        self.bert = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.arc_embedding = nn.Embedding(42, config.hidden_size)
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

        arc_texts = [RelationTable[i] for i in arcs.to('cpu').tolist()]

        input = self.tokenizer(arc_texts, padding=True, return_tensors='pt', add_special_tokens=True)

        input = {k:v.to(input_ids.device) for k,v in input.items()}
        
        arc_outputs = self.bert(**input)
        arc_outputs = arc_outputs.pooler_output
        
        final_embeddings = torch.cat([child_embeddings, head_embeddings, arc_outputs], dim=-1)

        return final_embeddings

    def forward(self, input_ids, input_attention_mask, token_ids, child_start_indices, child_end_indices, head_start_indices, head_end_indices, arcs):
        
        
        final_embeddings = self.get_embeddings(input_ids, input_attention_mask, token_ids, child_start_indices, child_end_indices, head_start_indices, head_end_indices, arcs) # n * dim

        logits_all = self.dep_label_classifier(final_embeddings) # n  * 2

        return logits_all
    



