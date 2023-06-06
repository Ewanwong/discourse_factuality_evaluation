
from my_model import RAEBaseModel
from my_model_data import set_random_seed, load_and_cache_examples
from transformers import AutoConfig, LongformerTokenizer, AdamW, get_scheduler
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def evaluate(model, eval_dataloader, loss_fn):
    losses = []
    all_sent_ids = []
    all_pos_probs = []
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(eval_dataloader):
            data = tuple([i.to(device) for i in data])
            sent_ids, input_ids, input_attention_mask, input_token_ids, child_start_indices, child_end_indices, head_start_indices, head_end_indices, dep_labels, arcs, sentence_label = data

            output = model(input_ids, input_attention_mask, input_token_ids, child_start_indices, child_end_indices, head_start_indices, head_end_indices, arcs)
            output /= 10
            loss = loss_fn(output, dep_labels)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("inf or nan in examples")
                continue
            
            losses.append(loss.item())

            _, pred = torch.max(output, dim=1)
            pos_probs = torch.nn.functional.softmax(output, dim=1)[:, 1]
            sent_ids, pred, labels, pos_probs = sent_ids.to('cpu').tolist(), pred.to('cpu').tolist(), dep_labels.to('cpu').tolist(), pos_probs.to('cpu').tolist()
            all_sent_ids += sent_ids
            all_preds += pred
            all_labels += labels
            all_pos_probs += pos_probs
    arc_acc, arc_balanced_acc, pos_acc, neg_acc = unbalanced_and_balanced_arc_accuracy(all_preds, all_labels)
    result_dict = unbalanced_and_balanced_sent_accuracy(all_sent_ids, all_preds, all_labels)
    avg_loss = sum(losses) / len(losses)
    #print(all_preds)
    #print(all_labels)
    model.train()
    return avg_loss, arc_acc, arc_balanced_acc, pos_acc, neg_acc, result_dict


def train(model, train_dataloader, eval_dataloader, test_dataloader, eval_save_path, test_save_path, log_file, optimizer, lr_scheduler, loss_fn, epochs, eval_steps):
    with open(log_file, 'w') as f:
        step = 0
        losses = []
        best_eval_acc = 0
        best_test_acc = 0
        for epoch in range(epochs):

            model.train()
            
            
            
            for data in tqdm(train_dataloader):
                data = tuple([i.to(device) for i in data])
                sent_ids, input_ids, input_attention_mask, input_token_ids, child_start_indices, child_end_indices, head_start_indices, head_end_indices, dep_labels, arcs, sentence_label = data
                
                #feature['input_tree'] = feature['input_tree'].requires_grad()
                #feature['context_tree'] = feature['context_tree'].requires_grad()
                
                
                output = model(input_ids, input_attention_mask, input_token_ids, child_start_indices, child_end_indices, head_start_indices, head_end_indices, arcs)
                output /= 10
                # print(dep_labels)
                loss = loss_fn(output, dep_labels)
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print("inf or nan in examples")
                    continue
                step  +=  1
                
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                    
                
                losses.append(loss.item())

                # loss.backward()

                
                
                if step % eval_steps == 0:
                
                    print("\n==========================================EVALUATION_RESULTS===============================")
                    print("training loss:")
                    print(sum(losses)/len(losses))

                    print("\n==========================================EVALUATION_RESULTS===============================", file=f)
                    print("training loss:", file=f)
                    print(sum(losses)/len(losses), file=f)

                    losses = []

                    avg_loss, arc_acc, arc_balanced_acc, pos_acc, neg_acc, result_dict = evaluate(model, eval_dataloader, loss_fn)
                    print("eval loss:")
                    print(avg_loss)
                    print("eval arc accuracy:")
                    print(arc_acc, arc_balanced_acc, pos_acc, neg_acc)
                    print("eval sentence accuracy:")
                    print(result_dict)
                    print("eval sentence best performance:")
                    print(max([result_dict[k][3] for k in result_dict.keys()]))

                    if max([result_dict[k][3] for k in result_dict.keys()]) > best_eval_acc:
                        best_eval_acc = max([result_dict[k][3] for k in result_dict.keys()])
                        model.save_pretrained(eval_save_path)

                    print("eval loss:", file=f)
                    print(avg_loss, file=f)
                    print("eval arc accuracy:", file=f)
                    print(arc_acc, arc_balanced_acc, pos_acc, neg_acc, file=f)
                    print("eval sentence accuracy:", file=f)
                    print(result_dict, file=f)
                    print("eval sentence best performance:", file=f)
                    print(max([result_dict[k][3] for k in result_dict.keys()]), file=f)


                    avg_loss, arc_acc, arc_balanced_acc, pos_acc, neg_acc, result_dict = evaluate(model, test_dataloader, loss_fn)
                    print("test loss:")
                    print(avg_loss)
                    print("test accuracy:")
                    print(arc_acc, arc_balanced_acc, pos_acc, neg_acc)
                    print("test sentence accuracy:")
                    print(result_dict)
                    print("test sentence best performance:")
                    print(max([result_dict[k][3] for k in result_dict.keys()]))

                    if max([result_dict[k][3] for k in result_dict.keys()]) > best_test_acc:
                        best_test_acc = max([result_dict[k][3] for k in result_dict.keys()])
                        model.save_pretrained(test_save_path)

                    print("test loss:", file=f)
                    print(avg_loss, file=f)
                    print("test accuracy:", file=f)
                    print(arc_acc, arc_balanced_acc, pos_acc, neg_acc, file=f)
                    print("test sentence accuracy:", file=f)
                    print(result_dict, file=f)
                    print("test sentence best performance:", file=f)
                    print(max([result_dict[k][3] for k in result_dict.keys()]), file=f)

                    print('best eval:')
                    print(best_eval_acc)
                    print('best test:')
                    print(best_test_acc)

                    print('best eval:', file=f)
                    print(best_eval_acc, file=f)
                    print('best test:', file=f)
                    print(best_test_acc, file=f)


                    

        print("\n==========================================Training_finished===============================")
        print("best eval performance:")
        print(best_eval_acc)
        print("best test performance:")
        print(best_test_acc)

        print("\n==========================================Training_finished===============================", file=f)
        print("best eval performance:", file=f)
        print(best_eval_acc, file=f)
        print("best test performance:", file=f)
        print(best_test_acc, file=f)
                    


def unbalanced_and_balanced_arc_accuracy(preds, labels):
    assert len(preds) == len(labels)
    total = len(preds)
    pos_total = 0
    pos_correct = 0
    correct = 0
    for pred, label in zip(preds, labels):
        if pred == label:
            correct += 1
        if label == 1:
            pos_total += 1
        if label == 1 and pred == label:
            pos_correct += 1
    acc = correct / total
    neg_total = total - pos_total
    neg_correct = correct - pos_correct
    pos_acc = pos_correct / pos_total
    neg_acc = neg_correct / neg_total
    balanced_acc = (pos_acc + neg_acc) / 2
    return acc, balanced_acc, pos_acc, neg_acc

def unbalanced_and_balanced_sent_accuracy(sent_ids, pos_probs, labels):
    assert len(pos_probs) == len(labels)
    assert len(sent_ids) == len(labels)

    pred_dict = {}

    for sent_id, pred, label in zip(sent_ids, pos_probs, labels):
        if sent_id not in pred_dict.keys():
            pred_dict[sent_id] = {}
            pred_dict[sent_id]['preds'] = []
            pred_dict[sent_id]['labels'] = []
        pred_dict[sent_id]['preds'].append(pred)
        pred_dict[sent_id]['labels'].append(label)
    
    result_dict = {}

    for i in range(1, 10):
        threshold = i / 10
        total = len(pred_dict.keys())
        correct = 0
        pos_correct = 0
        pos_total = 0
        for k, v in pred_dict.items():
            assert sum(v['labels'])==0 or sum(v['labels'])==len(v['labels'])
            pred = sum(v['preds']) / len(v['preds'])
            if pred >= threshold:
                pred = 1
            else:
                pred = 0
            if pred == v['labels'][0]:
                correct += 1
            if v['labels'][0] == 1:
                pos_total += 1
            if pred == v['labels'][0] and v['labels'][0] == 1:
                pos_correct += 1
        acc = correct / total
        neg_total = total - pos_total
        neg_correct = correct - pos_correct
        pos_acc = pos_correct / pos_total
        neg_acc = neg_correct / neg_total
        balanced_acc = (pos_acc + neg_acc) / 2

        result_dict[threshold] = (acc, pos_acc, neg_acc, balanced_acc)
    return result_dict


if __name__ == '__main__':

    set_random_seed(1)
    
    config = AutoConfig.from_pretrained('allenai/longformer-base-4096')
    model = RAEBaseModel(config)
    device = torch.device('cuda')
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

    data = 'cnndm'
    data_type = 'entity'

    train_data_path = f'../data/{data}_{data_type}_rst_parsed_train.json'
    train_feature_path = f'../data/my_model_{data}_{data_type}_rst_parsed_train.pt'
    eval_data_path = f'../data/{data}_{data_type}_rst_parsed_dev.json'
    eval_feature_path = f'../data/my_model_{data}_{data_type}_rst_parsed_dev.pt'
    test_data_path = f'../data/{data}_human_rst_parsed_test.json'
    test_feature_path = f'../data/my_model_{data}_human_rst_parsed_test.pt'

    eval_save_path = f'../{data}_eval_best_entity'
    test_save_path = f'../{data}_test_best_entity'

    log_file = f'{data}_baseline_{data_type}_lr_1e-6_bsz_8.txt'

    model = model.to(device)

    train_dataset = load_and_cache_examples(train_data_path, train_feature_path, 2048, tokenizer)
    eval_dataset = load_and_cache_examples(eval_data_path, eval_feature_path, 2048, tokenizer)
    test_dataset = load_and_cache_examples(test_data_path, test_feature_path, 2048, tokenizer)
    
    bsz = 8

    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=bsz, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=bsz, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=1e-6)
    loss_fn = nn.CrossEntropyLoss()

    train_num = len(train_dataset)

    lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=1500, num_training_steps=20 * train_num
        )

    train(model, train_dataloader, eval_dataloader, test_dataloader, eval_save_path, test_save_path, log_file, optimizer, lr_scheduler=lr_scheduler, loss_fn=loss_fn, epochs=20, eval_steps=200)
