from sentence_transformers import SentenceTransformer, models, InputExample, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch import nn
from torch.utils.data import DataLoader
import json
import os

from siamese_transformer_baseline_model import create_examples


if __name__ == '__main__':
    test_samples = create_examples('../data/cnndm_human_test.json')
    for steps in range(1000, 15000, 1000):
        checkpoint_path = f'../siamese_model/cnndm_entity_bsz_8_epochs_2/all_checkpoints/{steps}' 
        print(checkpoint_path)
        model = SentenceTransformer(checkpoint_path)
        with open(os.path.join(checkpoint_path, 'test_result.txt'), 'w') as f:
            sim_scores = []

            sentences1 = [example.texts[0] for example in test_samples]
            sentences2 = [example.texts[1] for example in test_samples]
            labels = [example.label for example in test_samples]
            
            
            embeddings1 = model.encode(sentences1, convert_to_tensor=True)
            embeddings2 = model.encode(sentences2, convert_to_tensor=True)  

            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            scores =  [cosine_scores[i][i] for i in range(len(sentences1))]

            for i in range(1, 10):
                threshold = i / 10
                preds = [1.0 if score>=threshold else 0.0 for score in scores]
                assert len(preds) == len(labels)
                acc = [1 if preds[i]==labels[i] else 0 for i in range(len(preds))]
                unbalanced_acc = sum(acc) / len(acc)
                
                pos_total = sum(labels)
                neg_total = len(labels) - pos_total

                pos_correct = sum([1 if preds[i]==labels[i] and labels[i]==1.0 else 0 for i in range(len(preds))])
                neg_correct = sum(acc) - pos_correct

                pos_acc = pos_correct / pos_total
                neg_acc = neg_correct / neg_total

                balanced_acc = (pos_acc + neg_acc) / 2
                print(f"threshold: {threshold}  acc: {balanced_acc}  pos acc: {pos_acc}  neg acc: {neg_acc}  balanced acc: {balanced_acc}")
                print(f"threshold: {threshold}  acc: {balanced_acc}  pos acc: {pos_acc}  neg acc: {neg_acc}  balanced acc: {balanced_acc}", file=f)
            
