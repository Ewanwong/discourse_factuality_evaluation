import json
from tqdm import tqdm

path = "../data/cnndm_entity_rst_train.json"
output_path = "../data/cnndm_entity_rst_parsed_train.json"

def parse_tree_format(path):
    with open(path, 'r') as f:
        data = json.load(f)

    #print(len(data))
    #print(data[0].keys())

    # 输入的类型:
    # 单个example: a dict, keys为input_tree, context tree
    # 每个tree: a dict, keys有input, context(内容文本)
    # i个关系:
    # dep_idxi [[head start idx, head end idx], [child start idx, child end idx]], idx均从第一个词开始算
    # dep_labeli (1/-1 arc是否是正例), depi (label类型文本), dep_wordsi [head文本, child文本], sentlabel (context是否是input的正例)

    all_examples = []

    for item in tqdm(data):
        if item['context_tree_parsing_pred'] == ['NONE'] or item['input_tree_parsing_pred'] == ['NONE']:
            continue
        new_input_tokenized = []
        new_context_tokenized = []

        input_id_to_id = {} # 原有tokenization到新tokenization的映射
        context_id_to_id = {}

        input_tokenized = item['input_tokenized']
        context_tokenized = item['context_tokenized']

        for i in range(len(input_tokenized)):

            if input_tokenized[i][0] == '▁': # 独立词
                new_input_tokenized.append(input_tokenized[i][1:])
            elif input_tokenized[i][0] != '▁': # 非独立词，与上个词拼接
                new_input_tokenized[-1] += input_tokenized[i]
            #elif input_tokenized[i][0] == '▁' and len(input_tokenized[i]) == 1: # 标点符号前的_
            #    pass
            
            input_id_to_id[i] = len(new_input_tokenized) - 1 
        
        for i in range(len(context_tokenized)):
            if context_tokenized[i][0] == '▁':
                new_context_tokenized.append(context_tokenized[i][1:])
            elif context_tokenized[i][0] != '▁':
                new_context_tokenized[-1] += context_tokenized[i]
            #elif context_tokenized[i][0] == '▁' and len(context_tokenized[i]) == 1:
            #    pass
            context_id_to_id[i] = len(new_context_tokenized) - 1
        
        input_id_to_id[-1] = -1
        context_id_to_id[-1] = -1

        example_dict = {}
        input_tree_dict = {}
        context_tree_dict = {}

        input = ' '.join(new_input_tokenized) # 空格分隔各个词
        context = ' '.join(new_context_tokenized)

        input_tree_dict['input'] = input
        input_tree_dict['context'] = context 
        context_tree_dict['input'] = input
        context_tree_dict['context'] = context

        input_tree_dict[f'sentlabel'] = item['sentlabel']
        context_tree_dict[f'sentlabel'] = item['sentlabel']

        input_tree_preds = item['input_tree_parsing_pred']
        context_tree_preds = item['context_tree_parsing_pred']
        input_seg = item['input_segmentation_pred'] 
        context_seg = item['context_segmentation_pred']
        
        input_seg.insert(0,-1)# 第一个分割处为0
        context_seg.insert(0,-1)

        input_len = len(new_input_tokenized)

        input_head_starts, input_head_ends, input_child_starts, input_child_ends, input_relations = separate_tree_preds(input_tree_preds)
        for i in range(len(input_child_starts)):
            
            # start和end均包含
            input_tree_dict[f'dep_idx{i}'] = [[input_id_to_id[input_seg[input_head_starts[i]-1]]+1, input_id_to_id[input_seg[input_head_ends[i]]]], [input_id_to_id[input_seg[input_child_starts[i]-1]]+1, input_id_to_id[input_seg[input_child_ends[i]]]]]
            input_tree_dict[f'dep_label{i}'] = item['sentlabel']
            input_tree_dict[f'dep{i}'] = input_relations[i]
            input_tree_dict[f'dep_words{i}'] = [' '.join(new_input_tokenized[input_id_to_id[input_seg[input_head_starts[i]-1]]+1:input_id_to_id[input_seg[input_head_ends[i]]]+1]), ' '.join(new_input_tokenized[input_id_to_id[input_seg[input_child_starts[i]-1]]+1:input_id_to_id[input_seg[input_child_ends[i]]]+1])]

        context_head_starts, context_head_ends, context_child_starts, context_child_ends, context_relations = separate_tree_preds(context_tree_preds)
        for i in range(len(context_child_starts)):
            context_tree_dict[f'dep_idx{i}'] = [context_id_to_id[context_seg[context_head_starts[i]-1]]+1, context_id_to_id[context_seg[context_head_ends[i]]], context_id_to_id[context_seg[context_child_starts[i]-1]]+1, context_id_to_id[context_seg[context_child_ends[i]]]]
            context_tree_dict[f'dep_label{i}'] = item['sentlabel']
            context_tree_dict[f'dep{i}'] = context_relations[i]
            context_tree_dict[f'dep_words{i}'] = [' '.join(new_context_tokenized[context_id_to_id[context_seg[context_head_starts[i]-1]]+1:context_id_to_id[context_seg[context_head_ends[i]]]+1]), ' '.join(new_context_tokenized[context_id_to_id[context_seg[context_child_starts[i]-1]]+1:context_id_to_id[context_seg[context_child_ends[i]]]+1])]
            context_tree_dict[f'dep_idx{i}'] = [[context_tree_dict[f'dep_idx{i}'][0]+input_len, context_tree_dict[f'dep_idx{i}'][1]+input_len], [context_tree_dict[f'dep_idx{i}'][2]+input_len, context_tree_dict[f'dep_idx{i}'][3]+input_len]]

        example_dict['input_tree'] = input_tree_dict
        example_dict['context_tree'] = context_tree_dict

        all_examples.append(example_dict)
    return all_examples




def separate_tree_preds(tree_preds):
    
    items = tree_preds[0].split(' ')
    
    head_starts, head_ends, child_starts, child_ends, relations = [], [], [], [], []
    for item in items:
        head_start, head_end, child_start, child_end, relation = parse_tree_item(item)
        head_starts.append(head_start)
        head_ends.append(head_end)
        child_starts.append(child_start)
        child_ends.append(child_end)
        relations.append(relation)
    return head_starts, head_ends, child_starts, child_ends, relations

def parse_tree_item(item):
    
    head, child = item.split(',')

    head_split = head.split(':')
    child_split = child.split(':')
    
    head_start = int(head_split[0][1])
    head_end = int(head_split[-1])

    child_start = int(child_split[0])
    child_end = int(child_split[-1][:-1])

    relation = head_split[1].split('=') + child_split[1].split('=')
    relation = ' '.join(relation)
    return head_start, head_end, child_start, child_end, relation    




if __name__ == "__main__":


    with open(output_path, 'w') as f:
        json.dump(parse_tree_format(path), f)
