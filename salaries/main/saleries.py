import torch
import torch.nn as nn
import pandas as pd 
import numpy as np
from torch.autograd import Variable

dataset = pd.read_csv('./data/saleries_data/salaries.csv')

x_temp = dataset.iloc[:, :-1].values
y_temp = dataset.iloc[:, 1:].values   # or [1]

X_train = torch.FloatTensor(x_temp)
Y_train = torch.FloatTensor(y_temp)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1,5)
        self.fc2 = torch.nn.Linear(5,1)
     
    def forward(self, x):
        x = self.fc1(x)
        y_pred = self.fc2(x)
        return y_pred

model = Model()

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    X_train = Variable(X_train)
    Y_train = Variable(Y_train)
    y_pred = model(X_train)

    loss = loss_func(y_pred, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(epoch % 50 == 0):
        print(epoch, loss)

test_exp = Variable(torch.FloatTensor([[6.0]]))
print("6 years exp, Salary is:", model(test_exp).data.item())

