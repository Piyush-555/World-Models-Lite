from torch import nn

class Controller(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.fc = nn.Linear(nodes, 3)
        
    def forward(self, x):
        return self.fc(x)
