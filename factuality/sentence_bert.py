from sentence_transformers import SentenceTransformer
model = SentenceTransformer('allenai/longformer-base-4096')

#Sentences we want to encode. Example:
sentence = ['This framework generates embeddings for each input sentence']


#Sentences are encoded by calling model.encode()
embedding = model.encode(sentence)
print(embedding)