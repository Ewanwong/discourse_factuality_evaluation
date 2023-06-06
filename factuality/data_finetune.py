from datasets import load_dataset
from transformers import AutoTokenizer, LongformerForMaskedLM, LongformerPreTrainedModel, LongformerModel, get_scheduler, AdamW, DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):
        if index < len(self.dataset1):
            return self.dataset1[index]
        else:
            return self.dataset2[index - len(self.dataset1)]

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)


def train(model, train_dataloader, eval_dataloader, epochs, optimizer, lr_scheduler, eval_steps):
    step =  0
    for epoch in range(epochs):
        losses = []
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            batch = apply_mask(batch)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
            
            loss = outputs.loss
            losses.append(loss.item())

            loss.backward()
            # update parameters
            
            optimizer.step()
            lr_scheduler.step()

            step += 1

            #if step % 100 == 0:
            #    print(f'train loss: {sum(losses)/len(losses)}')
            

            if step % eval_steps == 0:
                print(step)
                print(f'train loss: {sum(losses)/len(losses)}')
                #eval_loss = evaluate(model, eval_dataloader)
                #print(f'eval loss: {eval_loss}')
                model.save_pretrained(f"finetuned_longformers/cnndm_longformer_{step}")
        

def evaluate(model, eval_dataloader):
    model.eval()
    with torch.no_grad():
        losses = []
        for batch in tqdm(eval_dataloader):
            
            batch = apply_mask(batch)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
            
            loss = outputs.loss
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

        


def apply_mask(inputs):

    
    inputs['labels'] = inputs['input_ids'].detach().clone()
    rand = torch.rand(inputs['input_ids'].shape)
    # create mask array
    mask_arr = (rand < 0.15) * (inputs['input_ids'] != 0) * \
                    (inputs['input_ids'] != 1) * (inputs['input_ids'] != 2)
            
    selection = []

    for i in range(inputs['input_ids'].shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )

    for i in range(inputs['input_ids'].shape[0]):
        inputs['input_ids'][i, selection[i]] = 50264
    
    
    inputs['labels'].masked_fill_(inputs['input_ids'] != 50264, -100)

    return inputs

if __name__ == '__main__':
    

    model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

    if os.path.exists('cnndm_train.pt'):
        train_dataset = torch.load('cnndm_train.pt')
        eval_dataset = torch.load('cnndm_dev.pt')
    
    else:
        dataset = load_dataset("cnn_dailymail", '3.0.0')
        # print(dataset['train'][0])
        def tokenize_function_input(examples):
            return tokenizer(examples["article"], padding="max_length", truncation=True, return_tensors='pt')
        
        def tokenize_function_context(examples):
            return tokenizer(examples["highlights"], padding="max_length", truncation=True, return_tensors='pt')
        


        tokenized_input_datasets = dataset.map(tokenize_function_input, batched=True, load_from_cache_file=False, num_proc=30)
        tokenized_context_datasets = dataset.map(tokenize_function_context, batched=True, load_from_cache_file=False, num_proc=30)

        for split in ('train', 'validation', 'test'):
            tokenized_input_datasets[split].remove_columns(['article', 'highlights'])
            tokenized_input_datasets[split].set_format("torch")
            tokenized_context_datasets[split].remove_columns(['article', 'highlights'])
            tokenized_context_datasets[split].set_format("torch")


        train_dataset = CombinedDataset(tokenized_input_datasets['train'], tokenized_context_datasets['train'])
        eval_dataset = CombinedDataset(tokenized_input_datasets['validation'], tokenized_context_datasets['validation'])

        data_collector = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15, pad_to_multiple_of=512, return_tensors='pt')
        torch.save(train_dataset, 'cnndm_train.pt')
        torch.save(eval_dataset, 'cnndm_dev.pt')

    bsz = 2
    epochs = 2
    lr = 5e-5
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=bsz)
    eval_dataloader = DataLoader(eval_dataset, batch_size=bsz)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    

    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_training_steps=epochs * len(train_dataset), num_warmup_steps=1000)

    train(model, train_dataloader, eval_dataloader, 2, optimizer, lr_scheduler, 2000)
    model.save_pretrained("finetuned_longformers/cnndm_longformer")
