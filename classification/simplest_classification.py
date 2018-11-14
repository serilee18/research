import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn as nn
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import randint

train = []
labels = []
color = ['b','r','y','m']
label = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

for i in range(1000):
    category = randint(0,3)
    if category == 0:
        tmp1 = np.random.uniform(0.1, 1)
        tmp2 = np.random.uniform(0.1, 1)
    elif category == 1:
        tmp1 = np.random.uniform(-0.1, -1)
        tmp2 = np.random.uniform(0.1, 1)
    elif category == 2:
        tmp1 = np.random.uniform(0.1, 1)
        tmp2 = np.random.uniform(-0.1, -1)
    elif category == 3:
        tmp1 = np.random.uniform(-0.1, -1)
        tmp2 = np.random.uniform(-0.1, -1)
    labels.append(label[category])
    train.append([tmp1, tmp2])
    plt.scatter(tmp1, tmp2, c=color[category])

plt.show()
plt.savefig("./figure2.png")

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

nlabel = len(labels[0])

classifier = _classifier(nlabel)

optimizer = optim.Adam(classifier.parameters())
criterion = nn.MultiLabelSoftMarginLoss()

epochs = 10
for epoch in range(epochs):
    losses = []
    for i, sample in enumerate(train):
        inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
        labelsv = Variable(torch.FloatTensor(labels[i])).view(1, -1)
        output = classifier(inputv)
        loss = criterion(output, labelsv)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.mean())
    print('[%d/%d] Loss: %.3f' % (epoch+1, epochs, np.mean(losses)))

output = classifier(Variable(torch.FloatTensor([0.9, 0.9])).view(1, -1))
print(output)

#SAVE checkpoint
PATH = 'classifier.pth'
torch.save(classifier.state_dict(), PATH)

