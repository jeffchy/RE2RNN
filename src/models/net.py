
import torch
import torch.nn as nn
import torch.nn.functional as F

class DAN(nn.Module):

    def __init__(self,
                 emb_dim=100,
                 hidden_dim=256,
                 dp=0.3,):
        super(DAN, self).__init__()

        self.dropout1 = nn.Dropout(dp)
        self.bn1 = nn.BatchNorm1d(emb_dim)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dp)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(dp)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout4 = nn.Dropout(dp)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):


        x = x.mean(dim=1)

        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.bn3(x)
        x = self.fc3(x)
        x = self.dropout4(x)
        x = self.bn4(x)
        # 2layers for simple dataset
        return x



class CNN(nn.Module):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, embedding_dim, hidden_dim):
        super(CNN, self).__init__()


        in_channel = 1

        kernel_sizes = [7, 7, 5, 5]

        self.conv = nn.ModuleList([nn.Conv2d(in_channel, hidden_dim, (K, embedding_dim)) for K in kernel_sizes])

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(len(kernel_sizes) * hidden_dim, hidden_dim)

    def forward(self, x):
        """
        :param input_x: a list size having the number of batch_size elements with the same length
        :return: batch_size X num_aspects tensor
        """
        # Conv & max pool
        x = x.unsqueeze(1)  # dim: (batch_size, 1, max_seq_len, embedding_size)

        # turns to be a list: [ti : i \in kernel_sizes] where ti: tensor of dim([batch, num_kernels, max_seq_len-i+1])
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]

        # dim: [(batch_size, num_kernels), ...]*len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        # Dropout & output
        x = self.dropout(x)  # (batch_size,len(kernel_sizes)*num_kernels)
        x = self.fc(x)
        return x