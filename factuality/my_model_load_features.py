
RelationTable = ['Attribution_SN', 'Enablement_NS', 'Cause_SN', 'Cause_NN', 'Temporal_SN',
                     'Condition_NN', 'Cause_NS', 'Elaboration_NS', 'Background_NS',
                     'Topic-Comment_SN', 'Elaboration_SN', 'Evaluation_SN', 'Explanation_NN',
                     'TextualOrganization_NN', 'Background_SN', 'Contrast_NN', 'Evaluation_NS',
                     'Topic-Comment_NN', 'Condition_NS', 'Comparison_NS', 'Explanation_SN',
                     'Contrast_NS', 'Comparison_SN', 'Condition_SN', 'Summary_SN', 'Explanation_NS',
                     'Enablement_SN', 'Temporal_NN', 'Temporal_NS', 'Topic-Comment_NS',
                     'Manner-Means_NS', 'Same-Unit_NN', 'Summary_NS', 'Contrast_SN',
                     'Attribution_NS', 'Manner-Means_SN', 'Joint_NN', 'Comparison_NN', 'Evaluation_NN',
                     'Topic-Change_NN', 'Topic-Change_NS', 'Summary_NN', ]
arc_dict = {}
for i in range(len(RelationTable)):
    arc_dict[RelationTable[i]] = i




def convert_examples_to_features_longformer(example_dict, tokenizer, max_length=512, pad_token=0, pad_token_segment_id=None,
                                      ):
    """
    每次一个句子
    每个input/context tree返回一个该句子全部的arc的batch
    """
    features = {}
    for type in ['context_tree', 'input_tree']:    
        example = example_dict['context_tree']
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
        
        input_arcs = []
        
        num_dependencies = 0
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

            arc_label = example['dep' + str(i)].split()
            p1, r1, p2, r2 = arc_label
            if r1==r2:
                relation = r1
            elif r1=='span':
                relation = r2
            else:
                relation=r1
            rst_label = f'{relation}_{p1[0]}{p2[0]}'
            input_arcs.append(arc_dict[rst_label])
            

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
        
        input_ids = [input_ids for _ in range(num_dependencies)]
        input_attention_mask = [input_attention_mask for _ in range(num_dependencies)]
        token_ids = [token_ids for _ in range(num_dependencies)]

        
        features[type]=InputFeatures(input_ids=input_ids,
                                    input_attention_mask=input_attention_mask,
                                    token_ids=token_ids,
                                    sentence_label=sentence_label,
                                    child_start_indices=child_start_indices,
                                    child_end_indices=child_end_indices,
                                    head_start_indices=head_start_indices,
                                    head_end_indices=head_end_indices,
                                    dep_labels=dep_labels,
                                    arcs=input_arcs
                                    )
            
    
    
    return features


class InputFeatures(object):
    def __init__(self, input_ids, input_attention_mask, token_ids, sentence_label, child_start_indices, child_end_indices, head_start_indices, head_end_indices,
                 dep_labels, arcs):
        self.input_ids = input_ids
        self.input_attention_mask = input_attention_mask
        self.token_ids = token_ids
        self.sentence_label = sentence_label
        self.child_start_indices = child_start_indices
        self.child_end_indices = child_end_indices
        self.head_start_indices = head_start_indices
        self.head_end_indices = head_end_indices
        self.dep_labels = dep_labels
        self.arcs = arcs
        

def pad_1d(input, max_length, pad_token):
    padding_length = max_length - len(input)
    if padding_length < 0:
        input = input[:max_length]
        padding_length = 0
    input = input + ([pad_token] * padding_length)
    return input