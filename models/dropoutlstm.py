################################################################################
# This LSTM / Variational dropout code is part of                              #
# https://github.com/keitakurita/Better_LSTM_PyTorch                           #
# package which implements a version of recurrent dropout                      #
# similar to the Adrian Gal paper. The code is put as a file here              #
# as its seems that the package is no longer available to download via pip     #
################################################################################



import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from typing import *


class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout: float, batch_first: Optional[bool]=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x):
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

class DropoutLSTM(nn.Module):
    def __init__(self, hidden_dim, vocab_size, embedding_dim=300,batch_size=64,
                 weight_matrix=None,dvc=None,tr_embed=False, dropouti=0.,
                 dropoutw = 0., dropouto = 0.,
                 batch_first=False):
        super(DropoutLSTM, self).__init__()
        self.dropoutw = dropoutw

        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.load_state_dict({'weight':weight_matrix})
        self.embedding.weight.requires_grad = tr_embed
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 90)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.device = dvc

    def _drop_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                getattr(self.lstm, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()


    def forward(self, input):
        input = self.embedding(input)
        h_0 = torch.zeros(1, input.size()[1], self.hidden_dim).to(self.device)
        c_0 = torch.zeros(1, input.size()[1], self.hidden_dim).to(self.device)
        self._drop_weights()
        input = self.input_drop(input)
        out, _ = self.lstm.forward(input, (h_0, c_0))
        out = self.output_drop(out)[-1]
        return self.output_layer(out)