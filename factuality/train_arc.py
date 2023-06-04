from raemodel import RAEBaseModel, InputFeatures, convert_examples_to_features_longformer, load_and_cache_examples
from transformers import AutoConfig, LongformerTokenizer, AdamW, get_scheduler
import torch
import torch.nn as nn
from tqdm import tqdm



def evaluate(model, data_path, feature_path, max_length, tokenizer, shuffle):
    dataloader = load_and_cache_examples(data_path, feature_path, tokenizer=tokenizer, max_length=max_length, shuffle=shuffle)
    model.eval()
    with torch.no_grad():
        total = 0
        true_total = 0
        true_correct = 0
        correct = 0
        for data in dataloader:
            feature, label = data
        
            feature = {k: v.to(device) for k, v in feature.items()}
            label = label.to(device)

            output = model(feature, bsz=1)
            _, pred = torch.max(output, dim=1)
            
            pred = pred.tolist()
            label = label.tolist()

            if 0 in pred:
                final_pred = 0
            else:
                final_pred = 1
            
            if 0 in label:
                final_label = 0
            else:
                final_label = 1

            if final_pred == final_label:
                correct += 1
            total += 1
            if final_label == 1:
                true_total += 1
            if final_pred == final_label and final_label == 1 :
                true_correct += 1
        acc = correct / total
        true_acc = true_correct / true_total
        false_acc = (correct-true_correct) / (total - true_total)
        label_balanced_acc = (true_acc+false_acc) / 2
    return acc, label_balanced_acc

def train(model, optimizer, lr_scheduler, loss_fn, batch, epochs, data_path, feature_path, max_length, tokenizer, shuffle):
    step = 0
    batch_loss = torch.tensor(0.0).to(device)
    losses = []
    for epoch in range(epochs):
        train_dataloader = load_and_cache_examples(data_path=data_path, feature_path=feature_path, max_length=max_length, tokenizer=tokenizer, shuffle=shuffle)
        model.train()
        
        
        
        for data in tqdm(train_dataloader):
            feature, label = data
            
            feature = {k: v.to(device) for k, v in feature.items()}
            if label[0].item() == -1:
                continue
            #feature['input_tree'] = feature['input_tree'].requires_grad()
            #feature['context_tree'] = feature['context_tree'].requires_grad()
            label = label.to(device)
            
            output = model(feature, bsz=1)
            
            loss = loss_fn(output, label)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("inf or nan in examples")
                continue
            step  +=  1
            if (step+1) % batch == 0:
                batch_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0).to(device)
            else:
                batch_loss += loss
                
            
            losses.append(loss.item())

            # loss.backward()

            
            
            if (step + 1) % 2000 == 0:
            
                print("\n==========================================EVALUATION_RESULTS===============================")
                print("training loss:")
                print(sum(losses)/len(losses))
                acc, label_balanced_acc = evaluate(model, '../cnndm_generation_rst_parsed_dev.json', '../cnndm_generation_rst_parsed_dev.pt', max_length, tokenizer, shuffle)
                print("dev performance:")
                print(acc)
                print(label_balanced_acc)
                acc, label_balanced_acc = evaluate(model, '../cnndm_human_rst_parsed_test.json', '../cnndm_human_rst_parsed_test.pt', max_length, tokenizer, shuffle)
                print('test performance:')
                print(acc)
                print(label_balanced_acc)
        # print(evaluate(model, train_dataloader))
        


if __name__ == '__main__':
    
    config = AutoConfig.from_pretrained('allenai/longformer-base-4096')
    model = RAEBaseModel(config)
    device = torch.device('cuda')
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

    data_path = '../cnndm_generation_rst_parsed_train.json'
    feature_path = '../cnndm_generation_rst_parsed_train.pt'
    # dev_feature_path = '../xsum_generation_rst_parsed_dev_dev.pt'
    model = model.to(device)

    dataloader = load_and_cache_examples(data_path, feature_path, 2048, tokenizer, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    train_num = sum(1 for _ in dataloader)

    lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=4 * train_num
        )

    train(model, optimizer, lr_scheduler=lr_scheduler, loss_fn=loss_fn, batch=4, epochs=16, data_path='../cnndm_generation_rst_parsed_train.json', feature_path='../cnndm_generation_rst_parsed_train.pt', max_length=2048, tokenizer=tokenizer, shuffle=True)