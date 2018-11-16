import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable

class _classifier(nn.Module):
    def __init__(self, nlabel):
        super(_classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, nlabel),
        )

    def forward(self, input):
        return self.main(input)


model = _classifier(4)

# Load Checkpoint
PATH = 'classifier.pth'
model.load_state_dict(torch.load(PATH))


output = model(Variable(torch.FloatTensor([0.9, 0.9])).view(1, -1))
print(output)