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
    all_input_ids, input_attention_mask, input_token_ids, input_child_start_indices, input_child_end_indices, input_head_start_indices, input_head_end_indices, context_child_start_indices, context_child_end_indices, context_head_start_indices, context_head_end_indices, dep_labels, input_arcs, context_arcs, sentence_label = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    sent_ids = []
    context_arc_ids = []
    for id, f in enumerate(features):
        input_arc_num = len(f['input_tree'].dep_labels)
        context_arc_num = len(f['context_tree'].dep_labels)

        sent_ids += [id] * input_arc_num * context_arc_num

        all_input_ids += f['context_tree'].input_ids * input_arc_num
        input_attention_mask += f['context_tree'].input_attention_mask * input_arc_num
        input_token_ids += f['context_tree'].token_ids * input_arc_num
        dep_labels += f['context_tree'].dep_labels * input_arc_num
        sentence_label += f['context_tree'].sentence_label * input_arc_num
       


        input_child_start_indices += f['input_tree'].child_start_indices * context_arc_num
        input_child_end_indices += f['input_tree'].child_end_indices * context_arc_num
        input_head_start_indices += f['input_tree'].head_start_indices * context_arc_num
        input_head_end_indices += f['input_tree'].head_end_indices * context_arc_num              
        input_arcs += f['input_tree'].arcs * context_arc_num

        context_child_start_indices += f['context_tree'].child_start_indices * input_arc_num
        context_child_end_indices += f['context_tree'].child_end_indices * input_arc_num
        context_head_start_indices += f['context_tree'].head_start_indices * input_arc_num
        context_head_end_indices += f['context_tree'].head_end_indices * input_arc_num              
        context_arcs += f['context_tree'].arcs * input_arc_num

        context_arc_ids += [i for i in range(context_arc_num)] * input_arc_num
        


    sent_ids = torch.tensor(sent_ids)

    all_input_ids = torch.tensor(all_input_ids).squeeze(1)
    input_attention_mask = torch.tensor(input_attention_mask).squeeze(1)
    input_token_ids = torch.tensor(input_token_ids).squeeze(1)

    input_child_start_indices = torch.tensor(input_child_start_indices)
    input_child_end_indices = torch.tensor(input_child_end_indices)
    input_head_start_indices = torch.tensor(input_head_start_indices)
    input_head_end_indices = torch.tensor(input_head_end_indices)

    context_child_start_indices = torch.tensor(context_child_start_indices)
    context_child_end_indices = torch.tensor(context_child_end_indices)
    context_head_start_indices = torch.tensor(context_head_start_indices)
    context_head_end_indices = torch.tensor(context_head_end_indices)

    dep_labels = torch.tensor(dep_labels)
    
    input_arcs = torch.tensor(input_arcs)
    context_arcs = torch.tensor(context_arcs)
    
    sentence_label = torch.tensor(sentence_label)
    context_arc_ids = torch.tensor(context_arc_ids)


    dataset = TensorDataset(sent_ids, context_arc_ids, all_input_ids, input_attention_mask, input_token_ids, input_child_start_indices, input_child_end_indices, input_head_start_indices, input_head_end_indices, context_child_start_indices, context_child_end_indices, context_head_start_indices, context_head_end_indices,
                            dep_labels, input_arcs, context_arcs, sentence_label)

    return dataset


