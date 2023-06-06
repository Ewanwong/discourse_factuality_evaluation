from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch import nn
from torch.utils.data import DataLoader
import json


def create_examples(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = []
    for sent_dict in data:
        sent_dict = sent_dict['context_tree']
        for i in range(len(sent_dict.keys()-3)//4):
            samples.append(InputExample(texts='<s>'+[sent_dict['input']+'</s>', '<s>'+'</s></s>'.join(sent_dict['context'], sent_dict[f'dep_words{i}'][0], sent_dict[f'dep_words{i}'][1], sent_dict[f'dep{i}'])+'</s>'], label=float(sent_dict['sentlabel'])))
    
    return samples



if __name__ == "__main__":

    data_type = 'entity'
    #prepare dataset
    train_samples = create_examples(f'../data/cnndm_{data_type}_rst_parsed_train.json')
    dev_samples = create_examples(f'../data/cnndm_{data_type}_rst_parsed_dev.json')
    test_samples = create_examples('../data/cnndm_human_parsed_rst_test.json')

    # define a model
    word_embedding_model = models.Transformer('allenai/longformer-base-4096')
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # loss fct
    train_loss = losses.CosineSimilarityLoss(model=model)

    train_batch_size=8
    num_epochs=2

    model_save_path = f'../siamese_w_output_arcs_model/cnndm_{data_type}_bsz_8_epochs_2'
    checkpoint_path = f'../siamese_w_output_arcs_model/cnndm_{data_type}_bsz_8_epochs_2/all_checkpoints'
    warmup_steps = 1500

    # prepare dataset and dataloader
    
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)


    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='dev')

    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            scheduler='WarmupLinear',
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            save_best_model=True,
            show_progress_bar=True,
            checkpoint_save_steps=1000,
            checkpoint_path=checkpoint_path,)


    # evaluate on test set
    model = SentenceTransformer(model_save_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='test')
    test_evaluator(model, output_path=model_save_path)