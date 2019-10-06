import torch
import torch.nn as nn
class KimCNN(nn.Module):
    def __init__(self, embedding_dim, vocab_size, embed_matrix, max_size,
                 dvc=None):
        super(KimCNN, self).__init__()
        # Here we will implement the model from Kims paper
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.load_state_dict({'weight':embed_matrix})
        self.embedding_dim = embedding_dim
        self.max_size = max_size
        self.num_output_channels = 1
        self.dropout = nn.Dropout(0.5)

        self.first_pass = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=self.num_output_channels,
                      kernel_size=3,
                      stride=1),
            nn.MaxPool2d(4, 4),
            nn.ReLU(inplace=False)
        )

        self.second_pass = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=self.num_output_channels,
                      kernel_size=4,
                      stride=1),
            nn.MaxPool2d(4, 4),
            nn.ReLU(inplace=False)
        )

        self.third_pass = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=self.num_output_channels,
                      kernel_size=5,
                      stride=1),
            nn.MaxPool2d(4, 4),
            nn.ReLU(inplace=False)
        )
        self.l1 = nn.Linear(372*74*self.num_output_channels, 512)
        self.l2 = nn.Linear(512, 90)
        self.device = dvc

    def forward(self, x):
        kernel_outputs = []
        x = self.embeddings(x).permute(1, 0, 2).unsqueeze(1)
        padded_x = torch.zeros((x.shape[0], self.max_size, self.embedding_dim)).to(self.device)
        # here we could experiment with cutting the size of the padded
        # input to make the convolutions computationally managable
        s = min(x.shape[2]-1, self.max_size)
        x = x[:, :, :s, :]
        padded_x[:, :x.shape[2], :] = x.squeeze()
        padded_x = padded_x.unsqueeze(1)
        kernel_outputs.append(self.first_pass(padded_x))
        kernel_outputs.append(self.second_pass(padded_x))
        kernel_outputs.append(self.third_pass(padded_x))
        padded_x = torch.cat(kernel_outputs, dim=2)
        padded_x = self.l1(padded_x.view((padded_x.size(0), -1)))
        padded_x = self.l2(padded_x)
        return padded_x