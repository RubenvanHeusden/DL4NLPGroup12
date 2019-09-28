import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, hidden_dim, vocab_size, embedding_dim=300, batch_size=64,
                 weight_matrix=None, dvc=None, tr_embed=False):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.load_state_dict({'weight':weight_matrix})
        self.embedding.weight.requires_grad = tr_embed
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,bidirectional=True)
        self.output_layer = nn.Linear(2*hidden_dim, 90)
        self.hidden_dim = hidden_dim
        self.drop = nn.Dropout(0.3)
        self.device = dvc

    def forward(self, x):
        x = self.embedding(x)
        # maybe zeros or xavier / normal distribution
        h_0 = torch.zeros(2, x.size()[1], self.hidden_dim).to(self.device)
        c_0 = torch.zeros(2, x.size()[1], self.hidden_dim).to(self.device)
        h, _ = self.lstm(x, (h_0, c_0))
        # maybe use the 'stagger' trick here but not too sure about that
        # we can uncomment the code below to test it
        # forward_output, backward_output = h[:-1, :,
        #                                   :self.hidden_dim], h[1:, :,
        #                                                  self.hidden_dim:]
        # h = torch.cat((forward_output, backward_output), dim=-1)
        h = self.drop(h)[-1]
        out = self.output_layer(h)
        return out