import torch 
import torch.nn as nn
from torch.autograd import Variable 


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, window_size, num_layers=1, dropout_prob=0.5):
        super(LSTMModel, self).__init__()
        
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
        self.dropout2 = nn.Dropout(dropout_prob)
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (hn, cn) = self.lstm1(x)
        
        out = self.dropout1(out)
        
        out, (hn, cn) = self.lstm2(out)
        
        out = self.dropout2(out[:, -1, :])  
        
        out = self.fc(out)
        
        return out

  
    
if __name__ == '__main__':
    
    input_dim = 10        
    hidden_dim = 8        
    output_dim = 3        
    window_size = 20      
    dropout_prob = 0.5    

    model = LSTMModel(input_dim, hidden_dim, output_dim, window_size, dropout_prob=dropout_prob)

    x = torch.randn(32, window_size, input_dim)  # (batch_size=32, sequence_length=window_size, input_features=input_dim)

    output = model(x)
    print(output.shape)  
