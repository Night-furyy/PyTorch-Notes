import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd

df = pd.read_csv('./data/boston_housing_data/train.csv')

#data = data.apply(pd.to_numeric)
train_array = df.values
x_train = train_array[:,1:14]
y_train = train_array[:,14:]
print(y_train)
EPOCH = 501
HL = 20

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13,HL)
        self.fc2 = nn.Linear(HL,1)
    def forward(self, x):
        #x = x.view(x.size(0), -1)
        out = F.relu(self.fc1(x))
        y_pred = self.fc2(out)
        return y_pred 

model = Model()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(EPOCH):
    X = Variable(torch.from_numpy(x_train).float())
    Y = Variable(torch.from_numpy(y_train).float())
    pred_y = model(X)
    #print(Y.shape)
    #print(pred_y.shape)
    
    loss = criterion(pred_y, Y)
     
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch % 50 == 0):
        
        print('Epoch: {}\tLoss: {}'.format(epoch, loss.item()))


def test():
    model.eval()
    df_test = pd.read_csv('./data/boston_housing_data/test.csv', error_bad_lines=False)
    test_array = df_test.iloc[:, 1:14].values
    x_test = test_array

    X = Variable(torch.from_numpy(x_test).float())
    pred_y = model(X)
    pred_y = pred_y.data.numpy()
    print('Result: \n {}'.format(pred_y))

test()

