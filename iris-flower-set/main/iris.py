import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

train_data = pd.read_csv('./data/iris_data/Iris_train.csv')
train_data.loc[train_data['species'] == 'Iris-setosa', 'species'] = 0
train_data.loc[train_data['species'] == 'Iris-versicolor', 'species'] = 1
train_data.loc[train_data['species'] == 'Iris-virginica', 'species'] = 2
train_data = train_data.apply(pd.to_numeric)
train_array = train_data.values

x_train = train_array[:,:4]
y_train = train_array[:,4]

test_data = pd.read_csv('./data/iris_data/Iris_test.csv')
test_data.loc[test_data['species'] == 'Iris-setosa', 'species'] = 0
test_data.loc[test_data['species'] == 'Iris-versicolor', 'species'] = 1
test_data.loc[test_data['species'] == 'Iris-virginica', 'species'] = 2
test_data = test_data.apply(pd.to_numeric)
test_array = test_data.values

x_test = test_array[:,:4]
y_test = test_array[:,4]

X = Variable(torch.Tensor(x_test).float())
Y = torch.Tensor(y_test).long().numpy()

HL = 10
EPOCH = 500

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, HL)
        self.fc2 = nn.Linear(HL, 3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Model()

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(EPOCH):
    x = Variable(torch.Tensor(x_train).float())
    y = Variable(torch.Tensor(y_train).long())

    pred_y = model(x)
    loss = criterion(pred_y, y)  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        pred_y = model(X)
        result = torch.max(pred_y.data, 1)[1].data.numpy()
        accuracy = float((result==Y).astype(int).sum()) / float(Y.size) 
        print("Epoch: {}\t | Loss: {}\t | Accuracy: {:.4f}%".format(epoch, loss.item(), accuracy*100))

pred_y = model(X)
result = torch.max(pred_y.data, 1)[1].data.numpy()
#_, result = torch.max(pred_y.data, 1)
#print("pred  ", result.numpy())
print("Pred:   ", result)
print("Actual: ", Y)
accuracy = float((result==Y).astype(int).sum()) / float(Y.size)
print("Accuracy: {:.4f}%".format(accuracy*100))
