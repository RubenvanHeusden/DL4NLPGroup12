import torch
from torch import nn

class SimpleLSTM(nn.Module):
    def __init__(self, hidden_dim, vocab_size, embedding_dim=300,batch_size=64,weight_matrix=None):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.load_state_dict({'weight':weight_matrix})
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 90)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.drop = nn.Dropout(0.3)
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

    def forward(self, x):
        x = self.embedding(x)
        # maybe zeros or xavier / normal distribution
        h_0 = torch.zeros(1, x.size()[1], self.hidden_dim)
        c_0 = torch.zeros(1, x.size()[1], self.hidden_dim)
        h, _ = self.lstm(x, (h_0, c_0))
        h = self.drop(h)[-1]
        out = self.output_layer(h)
        return out
