import os

import gdown
import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=5, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(5, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(5, x.size(0), self.hidden_size).to(x.device)
        x = self.embedding(x)
        out, _ = self.lstm(x, (h0, c0))
        x = self.fc1(out[:, -1, :])
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def load_model(model, url="https://drive.google.com/file/d/1v8Q6enV6XDM_IqM7bDvp9aqHvmWu2ax7", map_location=torch.device('cpu')):
    output = "weights/EmailClassifier.pt"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    model.load_state_dict(torch.load(output, map_location=map_location))
    return model
