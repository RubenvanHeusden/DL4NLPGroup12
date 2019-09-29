import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, embedding_dim, vocab_size, weight_matrix, max_size,
                 dvc=None):
        super(SimpleCNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.load_state_dict({'weight':weight_matrix})
        self.embedding_dim = embedding_dim
        self.max_size = max_size
        self.num_output_channels = 10
        self.dropout = nn.Dropout(0.5)
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.num_output_channels,
                               kernel_size=5,
                               stride=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(220*74*self.num_output_channels, 512)
        self.l2 = nn.Linear(512, 90)
        self.device = dvc

    def forward(self, x):
        x = self.embeddings(x).permute(1, 0, 2).unsqueeze(1)
        padded_x = torch.zeros((x.shape[0], self.max_size, self.embedding_dim)).to(self.device)
        # here we could experiment with cutting the size of the padded
        # input to make the convolutions computationally managable
        padded_x[:, :x.shape[2], :] = x.squeeze()
        padded_x = self.conv1(padded_x.unsqueeze(1))
        padded_x = self.relu(padded_x).squeeze()
        padded_x = self.pool(padded_x)
        padded_x = self.dropout(padded_x)
        padded_x = self.l1(padded_x.view(padded_x.size(0), -1))
        padded_x = self.l2(padded_x)
        return padded_x
