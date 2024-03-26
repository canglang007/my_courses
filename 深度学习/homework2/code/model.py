import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

    
# class RNN(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_size):
#         super(RNN, self).__init__()
#         self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
#         self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1)
#         self.fc1 = nn.Linear(hidden_size, 64)
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(64, 2)

#     def forward(self, x):
#         embedded = self.embedding(x)
#         lstm_out, _ = self.lstm(embedded)
        
#         # Global Max Pooling
#         max_pool, _ = torch.max(lstm_out, 1)
        
#         x = torch.relu(self.fc1(max_pool))
#         x = self.dropout(x)
#         x = self.fc2(x)
        
#         return torch.sigmoid(x)

class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim=512, hidden_size=128, num_layers=1, n_class=2,
                 bidirectional=False) -> None:
        super(RNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if bidirectional:
            self.D = 2
        else:
            self.D = 1
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=bidirectional)
        self.fc = nn.Linear(self.D * hidden_size, n_class)

    def forward(self, x):
        # x: [batch, max_len]
        batch_size = x.shape[0]
        output = self.embed(x)  # [b, max_len, 512]
        output, _ = self.rnn(output.transpose(0, 1))
        
        output = output[-1]
        output = self.fc(output)
        return output


class LSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim=512, hidden_size=128, num_layers=1, n_class=2,
                 bidirectional=False) -> None:
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if bidirectional:
            self.D = 2
        else:
            self.D = 1
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(self.D * hidden_size, n_class)

    def forward(self, x):
        # x: [batch, max_len]
        batch_size = x.shape[0]
        output = self.embed(x)  # [b, max_len, 512]
        output, (_, _) = self.lstm(output.transpose(0, 1))
        output = output[-1]
        output = self.fc(output)
        return output


class GRU(nn.Module):

    def __init__(self, vocab_size, embedding_dim=512, hidden_size=128, num_layers=1, n_class=2,
                 bidirectional=False) -> None:
        super(GRU, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if bidirectional:
            self.D = 2
        else:
            self.D = 1
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=bidirectional)
        self.fc = nn.Linear(self.D * hidden_size, n_class)

    def forward(self, x):

        batch_size = x.shape[0]
        output = self.embed(x) 
        output, _ = self.gru(output.transpose(0, 1))    # 交换张量的维度
        output = output[-1]
        output = self.fc(output)
        return output
if __name__ == '__main__':
    vocab_size = 1000
    seq_length = 500
    batch_size = 32
    
    input_data = torch.randint(1, vocab_size, (batch_size, seq_length))
    labels = torch.randint(0, 2, (batch_size, 1)).float()  # 二分类任务，0或1

    model = RNN(vocab_size=vocab_size, embedding_dim=100, hidden_size=128)

    output = model(input_data)

    print("Output shape:", output.shape)

    # 创建一个SummaryWriter来记录模型结构
    writer = SummaryWriter('./logs/model')
    writer.add_graph(model, input_data)
    writer.close()