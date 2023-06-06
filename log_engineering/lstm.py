import torch
from torch import nn
from torch.nn import functional as F

rnn = nn.LSTM(10, 20, bidirectional=True, batch_first=True)
input = torch.randn(2, 3, 10)
output, (hn, cn) = rnn(input)
output.shape


class LSTMEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.vocab_size = kwargs["vocab_size"]
        self.embedding_dim = kwargs["embedding_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.num_layers = kwargs["num_layers"]
        self.dropout = kwargs["dropout"]
        self.bidirectional = kwargs["bidirectional"]
        self.batch_first = kwargs["batch_first"]

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=self.batch_first,
        )

    def forward(self, x):
        out = self.embedding(x)
        out, hidden = self.lstm(out)
        hidden = [h.detach() for h in hidden]
        return out, hidden


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = kwargs["hidden_dim"]
        self.num_layers = kwargs["num_layers"]
        self.bidirectional = kwargs["bidirectional"]

        if self.bidirectional:
            self.fc1 = nn.Linear(self.hidden_dim * 2, 128)
        else:
            self.fc1 = nn.Linear(self.hidden_dim, 128)

        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class DNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = LSTMEncoder(args=args, kwargs=kwargs)
        self.decoder = Decoder(args=args, kwargs=kwargs)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
