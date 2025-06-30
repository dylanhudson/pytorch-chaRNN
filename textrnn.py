import torch
import torch.nn as nn

class TextRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size=128, num_layers=2):
        super(TextRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)

        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)

        output = output.reshape(-1, self.hidden_size)
        output = self.fc(output)        

        return output, hidden 
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)
    
    def export_model(self, path):
        torch.save(self.state_dict(), path)
        print(f'Model exported to {path}')

    #alternate constructor to load a model from a saved state dict back into a TextRNN object.
    @classmethod
    def load_model(cls, path, vocab_size, hidden_size=128, num_layers=2):
        model = cls(vocab_size, hidden_size, num_layers)
        model.load_state_dict(torch.load(path))
        print(f'Model loaded from {path}')
        return model
    


