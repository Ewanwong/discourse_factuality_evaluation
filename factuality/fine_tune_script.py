import logging
import os
import math
import copy
import torch
from dataclasses import dataclass, field
from transformers import AutoTokenizer, LongformerForMaskedLM, LongformerPreTrainedModel, LongformerModel, get_scheduler, AdamW, DataCollatorForLanguageModeling, Trainer, TextDataset
from transformers import TrainingArguments, HfArgumentParser


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



def pretrain_and_evaluate(args, model, tokenizer, eval_only, model_path):
    val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=args.val_datapath,
                              block_size=4096)
    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(f'Loading and tokenizing training data is usually slow: {args.train_datapath}')
        train_dataset = TextDataset(tokenizer=tokenizer,
                                    file_path=args.train_datapath,
                                    block_size=4096)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    trainer = Trainer(model=model, args=args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset)

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']
    logger.info(f'Initial eval bpc: {eval_loss/math.log(2)}')
    
    if not eval_only:
        trainer.train(model_path=model_path)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logger.info(f'Eval bpc after pretraining: {eval_loss/math.log(2)}')

if __name__ == '__main__':

    @dataclass
    class ModelArgs:
        attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
        max_pos: int = field(default=4096, metadata={"help": "Maximum position"})




    training_args=TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir='pre_trained_longformers',
        warmup_steps=500,
        learning_rate=0.00003,
        weight_decay=0.01,
        adam_epsilon=1e-6,
        max_steps=20000,
        logging_steps=500,
        save_steps=500,
        max_grad_norm=5.0,
        per_gpu_eval_batch_size=8,
        per_gpu_train_batch_size=2,  # 32GB gpu with fp32
        gradient_accumulation_steps=32,
        prediction_loss_only=True,
    )
    training_args.val_datapath = '../train_data.txt'
    training_args.train_datapath = '../validation_data.txt'

    model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

    pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path=None)