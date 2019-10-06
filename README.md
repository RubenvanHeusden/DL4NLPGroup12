# DL4NLP

The task of this project is to classify a text document into one of 90 different categories, and compare the performance between LSTMs and CNNs. We use two versions of either, a vanilla version and an augmented one.
We use the Reuters dataset, which has approximately 10000 documents to train on.
We preprocess the documents by filtering on a set of stopwords that is provided with the data.

Requires modules: 'torch', 'torchtext', 'sklearn', 'nltk'

There seems to be a small bug in the 'torchtext' module:
change '<python install folder>\Lib\site-packages\torchtext\utils.py' line 130 'csv.field_size_limit(sys.maxsize)' to 'csv.field_size_limit(maxInt)'


Usage: 'main.py -model -batch_size -n_epochs -use_cuda -hidden_dim'
('main.py -h' for an overview of the arguments)

We use the GLoVe pretrained embeddings with 300 dimensions to convert the words of each document into latent space.

**Used models:**
# (Simple)LSTM
Uses a vanilla LSTM model to process each document.

# BiLSTM

# (Simple)CNN
Uses a vanilla CNN model to process each document. Here, each document is padded or truncated to a fixed size.

# KimCNN
