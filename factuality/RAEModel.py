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
import sys
#import utils
#sys.modules['train_importance_utils'] = utils

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RAEModel(LongformerPreTrainedModel):
    def __init__(self, config):
        super(RAEModel, self).__init__(config)
        self.bert = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dep_label_classifier = nn.Linear(2 * 3 * config.hidden_size, 2)


        # self.init_weights()

    def get_embeddings(self, tree, bsz, device='cuda'):
        
        batch_size = tree.num_dependencies.item()
        
        transformer_outputs = self.bert(tree.input_ids[:1,...], attention_mask=tree.input_attention_mask[:1,...], token_type_ids=tree.token_ids[:1,...])

        outputs = transformer_outputs[0]
        outputs = self.dropout(outputs)
        outputs = outputs.view((-1, outputs.size(-1))) # bsz*len, emb_dim
        """
        add = torch.arange(batch_size) * tree.input_ids.size(1) 
        add = add.unsqueeze(1).to(device) #bsz*1

        child_start_temp = tree.child_start_indices + add
        child_end_temp = tree.child_end_indices + add
        head_start_temp = tree.head_start_indices + add
        head_end_temp = tree.head_end_indices + add
        """
        child_embeddings = torch.stack([torch.mean(outputs[tree.child_start_indices[i,].item():tree.child_end_indices[i,].item(),:], dim=0) for i in range(batch_size)], dim=0)
        head_embeddings = torch.stack([torch.mean(outputs[tree.head_start_indices[i,].item():tree.head_end_indices[i,].item():], dim=0) for i in range(batch_size)], dim=0) # bsz * emb_dim

        # child_embeddings = outputs[child_temp] 
        # head_embeddings = outputs[head_temp]

        child_embeddings = child_embeddings.view(batch_size, -1, child_embeddings.size(-1))
        head_embeddings = head_embeddings.view(batch_size, -1, head_embeddings.size(-1))
        
        arcs = tree.arcs.view(-1, tree.arcs.size(-1)) # bsz * len
        
        arc_outputs = []
        
        for i in range(int(batch_size/bsz)+1):
            
            if i * bsz >= batch_size:
                break
            elif (i+1) * bsz >= batch_size:
                batch_arcs = arcs[i*bsz:,...]
                batch_arc_label_lengths = tree.arc_label_lengths[i*bsz:,...]
            else:
                batch_arcs = arcs[i*bsz:(i+1)*bsz,...]
                batch_arc_label_lengths = tree.arc_label_lengths[i*bsz:(i+1)*bsz,...]
        
            
            
            # arc_label_lengths = tree.arc_label_lengths.view(-1) # bsz, 
            batch_arc_attention = torch.arange(batch_arcs.size(1)).to(device)[None, :] <= batch_arc_label_lengths[:, None] 
            batch_arc_attention = batch_arc_attention.type(torch.float)
            batch_arc_outputs = checkpoint(self.bert, batch_arcs, batch_arc_attention)
            batch_arc_outputs = batch_arc_outputs[1]
            batch_arc_outputs = self.dropout(batch_arc_outputs)
            
            arc_outputs.append(batch_arc_outputs)
        arc_outputs = torch.cat(arc_outputs, dim=0)
        
        arc_outputs = arc_outputs.view(batch_size, -1, arc_outputs.size(-1))  # bsz * 1 * emb_dim
       
        final_embeddings = torch.cat([child_embeddings, head_embeddings, arc_outputs], dim=2)

        return final_embeddings
    
    def get_both_embeddings(self, input_tree, context_tree, bsz, device='cuda'):
        
        input_embeddings =self.get_embeddings(input_tree, bsz, device) # m * 1 * dim
        context_embeddings = self.get_embeddings(context_tree, bsz, device=device) # n * 1 * dim
        input_arc_num = input_embeddings.shape[0]
        context_arc_num = context_embeddings.shape[0]
        input_embeddings = input_embeddings.unsqueeze(0).expand(context_arc_num, -1, -1, -1)
        context_embeddings = context_embeddings.unsqueeze(1).expand(-1, input_arc_num, -1, -1)
        assert context_embeddings.shape == input_embeddings.shape

        final_embeddings = torch.cat([input_embeddings, context_embeddings], dim=-1) # n * m * 1 * 2dim
        return final_embeddings
    
    def forward(self, features, bsz=4, device='cuda'):
        
        input_tree = features['input_tree']
        context_tree = features['context_tree']
        
        final_embeddings = self.get_both_embeddings(input_tree, context_tree, bsz, device) # n * m * 1 * 2dim

        logits_all = self.dep_label_classifier(final_embeddings) # n * m  * 1 * 2

        each_context_arc_scores, _ = torch.max(logits_all, dim=1) # n * 1 *  2

        example_avg_score = torch.mean(each_context_arc_scores, dim=0) # 1 * 2

        return example_avg_score
    
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(logits_all.view(-1, 2), context_arc.dep_labels.view(-1))

        outputs_return = (logits_all,)
        outputs_return = (loss,) + outputs_return

        return outputs_return




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


def pad_1d(input, max_length, pad_token):
    padding_length = max_length - len(input)
    if padding_length < 0:
        input = input[:max_length]
        padding_length = 0
    input = input + ([pad_token] * padding_length)
    return input


class InputFeatures(object):
    def __init__(self, input_ids, input_attention_mask, sentence_label, child_start_indices, child_end_indices, head_start_indices, head_end_indices,
                 dep_labels, num_dependencies, arcs, arc_labels, arc_label_lengths, token_ids=None):
        self.input_ids = input_ids
        self.input_attention_mask = input_attention_mask
        self.token_ids = token_ids
        self.sentence_label = sentence_label
        self.child_start_indices = child_start_indices
        self.child_end_indices = child_end_indices
        self.head_start_indices = head_start_indices
        self.head_end_indices = head_end_indices
        self.dep_labels = dep_labels
        self.num_dependencies = torch.tensor(num_dependencies)
        self.arcs = arcs
        self.arc_labels = arc_labels
        self.arc_label_lengths = arc_label_lengths
    
    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.input_attention_mask = self.input_attention_mask.to(device)
        self.token_ids = self.token_ids.to(device)
        self.sentence_label = self.sentence_label.to(device)
        self.child_start_indices = self.child_start_indices.to(device)
        self.child_end_indices = self.child_end_indices.to(device)
        self.head_start_indices = self.head_start_indices.to(device)
        self.head_end_indices = self.head_end_indices.to(device)
        self.dep_labels = self.dep_labels.to(device)
        self.num_dependencies = torch.tensor(self.num_dependencies.to(device))
        self.arcs = self.arcs.to(device)
        self.arc_labels = self.arc_labels.to(device)
        self.arc_label_lengths = self.arc_label_lengths.to(device)
        return self
    


# 输入的类型:
# 单个example: a dict, keys为input_tree, context tree
# 每个tree: a dict, keys有input, context(内容文本)
# i个关系:
# dep_idxi [[head start idx, head end idx], [child start idx, child end idx]], idx均从第一个词开始算
# dep_labeli (1/-1 arc是否是正例), depi (label类型文本), dep_wordsi [head文本, child文本], sentlabel (context是否是input的正例)

def convert_examples_to_features_longformer(example_dict, tokenizer, max_length=512, pad_token=0, pad_token_segment_id=None,
                                      ):
    """
    每次一个句子
    每个input/context tree返回一个该句子全部的arc的batch
    """


    features = {} # each instance has own features
    for type in ['input_tree', 'context_tree']:
        
        example = example_dict[type]
        index_map = {}

        input_inputs = tokenizer.encode_plus(' '+example['input'], add_special_tokens=True, return_token_type_ids=True)
        input_input_ids = input_inputs['input_ids']
        input_input_words = tokenizer.convert_ids_to_tokens(input_input_ids)

        context_inputs = tokenizer.encode_plus(' '+example['context'], add_special_tokens=True, return_token_type_ids=True)
        context_input_ids = context_inputs['input_ids']
        context_input_words = tokenizer.convert_ids_to_tokens(context_input_ids)

        input_word_start_positions = [i for i in range(len(input_input_ids)) if input_input_words[i][0] == 'Ġ']
        context_word_start_positions = [i+len(input_input_ids) for i in range(len(context_input_ids)) if context_input_words[i][0] == 'Ġ']
        word_start_positions = input_word_start_positions + context_word_start_positions


        assert len(word_start_positions) == len(example['input'].split(' ')) + len(example['context'].split(' '))

        for i in range(len(word_start_positions)):
            index_map[i] = word_start_positions[i]
        
        
        inputs = tokenizer.encode_plus(' '+example['input'], ' '+example['context'], add_special_tokens=True, return_token_type_ids=True)
        input_ids = inputs['input_ids']
        input_words = tokenizer.convert_ids_to_tokens(input_ids)
        assert len(input_ids) == len(input_words)
        token_ids = inputs['token_type_ids']

        index_map[len(word_start_positions)] = len(input_ids) - 1 # for boundary

        child_start_indices = []
        child_end_indices = []

        head_start_indices = []
        head_end_indices = []

        dep_labels = []
        num_dependencies = 0

        input_arcs = []
        arc_labels = []
        arc_label_lengths = []
        
        for i in range((len(example.keys())-3)//4):
            if len(example['dep_idx' + str(i)]) == 0:
                continue

            head_idx, child_idx = example['dep_idx' + str(i)][0], example['dep_idx' + str(i)][1]

            child_start_idx, child_end_idx = int(child_idx[0]), int(child_idx[1])
            head_start_idx, head_end_idx = int(head_idx[0]), int(head_idx[1])

            num_dependencies += 1
            dep_labels.append(int(example['dep_label' + str(i)]))
            
            child_start_indices.append(index_map[child_start_idx])
            child_end_indices.append(index_map[child_end_idx+1]) #TODO: boundary check
            head_start_indices.append(index_map[head_start_idx])
            head_end_indices.append(index_map[head_end_idx+1]) #TODO: boundary check

            arc_label_ids = tokenizer.encode(example['dep' + str(i)])
            arc_label_lengths.append(len(arc_label_ids))
            arc_labels.append(pad_1d(arc_label_ids, 20, pad_token))

            w1 = example['dep_words' + str(i)][0] # a sequence
            w2 = example['dep_words' + str(i)][1] # a sequence
            arc_text = '<>s' + example['dep' + str(i)] + ' </s></s> ' + w1 + ' </s></s> ' + w2 + '</s>'
            arc = tokenizer.encode(arc_text)
            input_arcs.append(pad_1d(arc, max_length, pad_token)) # TODO: decide pad length

        if len(input_ids) > max_length:
            
            return None
            # tokens_input = tokens_input[:max_length]
            

        if num_dependencies == 0:
            return None

        sentence_label = [int(example['sentlabel'])] * num_dependencies

        
        padding_length_a = max_length - len(input_ids)
        input_attention_mask = [[1] * len(input_ids) + [0] * padding_length_a]
        input_ids = [input_ids + ([pad_token] * padding_length_a)] 
        token_ids = [token_ids + ([0] * padding_length_a)] 
        
        
        
        """
        print(input_attention_mask)
        print(token_ids)
        print(sentence_label)
        print(head_start_indices)
        print(dep_labels)
        print(num_dependencies)
        print(input_arcs)
        print(arc_labels)
        print(arc_label_lengths)
        """

        def get_tensor(x):
            # print(x)
            assert isinstance(x, list) or isinstance(x, int)
            if torch.tensor(x).dim() < 2:
                return torch.tensor(x).unsqueeze(-1)
            else:
                return torch.tensor(x)
        # print(get_tensor(input_ids).shape)
        
        features[type]=InputFeatures(input_ids=get_tensor(input_ids),
                                        input_attention_mask=get_tensor(input_attention_mask),
                                        token_ids=get_tensor(token_ids),
                                        sentence_label=get_tensor(sentence_label),
                                        child_start_indices=get_tensor(child_start_indices),
                                        child_end_indices=get_tensor(child_end_indices),
                                        head_start_indices=get_tensor(head_start_indices),
                                        head_end_indices=get_tensor(head_end_indices),
                                        dep_labels=get_tensor(dep_labels),
                                        num_dependencies=num_dependencies,
                                        arcs=get_tensor(input_arcs),
                                        arc_labels=get_tensor(arc_labels),
                                        arc_label_lengths=get_tensor(arc_label_lengths))
            
        
    
    return features


def load_and_cache_examples(data_path, feature_path, max_length, tokenizer, shuffle):

    
    if os.path.exists(feature_path):
        logger.info("Loading features from cached file %s", feature_path)
        features = torch.load(feature_path)
    else:
        logger.info("Creating features from dataset file at %s", feature_path)
        with open(data_path, 'r') as f:
            examples = json.load(f)

            #examples = examples[:200]

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

    if shuffle:
        random.shuffle(features)
    
    for feature in features:
        if feature is None:
            continue
        yield feature, feature['input_tree'].sentence_label[0,:]


    
