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
from my_model_load_features import InputFeatures, convert_examples_to_features_longformer
logger = logging.getLogger(__name__)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def load_and_cache_examples(data_path, feature_path, max_length, tokenizer):

    
    if os.path.exists(feature_path):
        logger.info("Loading features from cached file %s", feature_path)
        features = torch.load(feature_path)
    else:
        logger.info("Creating features from dataset file at %s", feature_path)
        with open(data_path, 'r') as f:
            examples = json.load(f)


        features = []
        for example in tqdm(examples):
            
            feature = convert_examples_to_features_longformer(
                example,
                tokenizer,
                max_length=max_length,
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                
            )
            features.append(feature)
        logger.info("Saving features into cached file %s", feature_path)
        torch.save(features, feature_path)
    
    features = [f for f in features if f is not None]
    features = [f['context_tree'] for f in features if f['context_tree'].sentence_label[0]!=-1]
    all_input_ids, input_attention_mask, input_token_ids, child_start_indices, child_end_indices, head_start_indices, head_end_indices, dep_labels, arcs, sentence_label = [], [], [], [], [], [], [], [], [], []
    sent_ids = []
    for id, f in enumerate(features):
        sent_ids += [id for _ in range(len(f.dep_labels))]
        all_input_ids += f.input_ids
        input_attention_mask += f.input_attention_mask
        input_token_ids += f.token_ids

        child_start_indices += f.child_start_indices
        child_end_indices += f.child_end_indices
        head_start_indices += f.head_start_indices
        head_end_indices += f.head_end_indices

        dep_labels += f.dep_labels
        arcs += f.arcs
        
        sentence_label += f.sentence_label

    sent_ids = torch.tensor(sent_ids)

    all_input_ids = torch.tensor(all_input_ids).squeeze(1)
    input_attention_mask = torch.tensor(input_attention_mask).squeeze(1)
    input_token_ids = torch.tensor(input_token_ids).squeeze(1)

    child_start_indices = torch.tensor(child_start_indices)
    child_end_indices = torch.tensor(child_end_indices)
    head_start_indices = torch.tensor(head_start_indices)
    head_end_indices = torch.tensor(head_end_indices)

    dep_labels = torch.tensor(dep_labels)
    
    arcs = torch.tensor(arcs)
    
    sentence_label = torch.tensor(sentence_label)


    dataset = TensorDataset(sent_ids, all_input_ids, input_attention_mask, input_token_ids, child_start_indices, child_end_indices, head_start_indices, head_end_indices,
                            dep_labels, arcs, sentence_label)

    return dataset


