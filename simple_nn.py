from reutersdataset import ReutersDataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gloveembeddings import GloveEmbeddings
from sklearn.metrics import f1_score
from tqdm import tqdm

reuters_dataset = ReutersDataset('../reuters')
glove_embeddings = GloveEmbeddings(300)
glove_embeddings.load_embeddings('../glove.6B/glove.6B.300d.txt')
# now we can call glove_embeddings.embed_matrix and load this into the embed
# matrix of a pytorch network
glove_embeddings.set_embedding_matrix(reuters_dataset.vocab_dict)

