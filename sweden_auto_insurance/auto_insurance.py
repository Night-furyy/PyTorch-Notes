import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd

df = pd.read_csv('./data/auto_insurance_data/data.csv')

#data = data.apply(pd.to_numeric)
x_temp = df.iloc[:, :-1].values
y_temp = df.iloc[:, 1:].values

X_train = torch.FloatTensor(x_temp)
Y_train = torch.FloatTensor(y_temp)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,5)
        self.fc2 = nn.Linear(5,1)
    def forward(self, x):
        #x = x.view(x.size(0), -1)
        x = self.fc1(x)
        y_pred = self.fc2(x)
        return y_pred 

model = Model()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    #X = Variable(torch.from_numpy(x_train).float())
    #Y = Variable(torch.from_numpy(y_train).float())
    X_train = Variable(X_train)
    Y_train = Variable(Y_train)
    y_pred = model(X_train)
    #print(Y.shape)
    #print(pred_y.shape)
    
    loss = criterion(y_pred, Y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch % 50 == 0):
        
        print('Epoch: {}\tLoss: {}'.format(epoch, loss))

def test():
    model.eval()
    df_test = pd.read_csv('./data/auto_insurance_data/data.csv')
    test_array = df_test.values
    x_test = test_array[:, :-1]

    X = Variable(torch.FloatTensor(x_test))
    y_pred = model(X)
    y_pred = y_pred.data.numpy()
    print('Result: \n {}'.format(y_pred))

test()
