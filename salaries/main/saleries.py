import torch
import torch.nn as nn
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

dataset = pd.read_csv('./data/saleries_data/salaries.csv')

x_temp = dataset.iloc[:, :-1].values
y_temp = dataset.iloc[:, 1:].values

X_train = torch.FloatTensor(x_temp)
Y_train = torch.FloatTensor(y_temp)

#### MODEL ARCHITECTURE #### 

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,5)
        self.lin2 = torch.nn.Linear(5,1)
     
    def forward(self, x):
        x = self.linear(x)
        y_pred = self.lin2(x)
        return y_pred

model = Model()

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#print(len(list(model.parameters())))
def count_params(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

### TRAINING 
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
    #count = count_params(model)
    #print(count)

test_exp = Variable(torch.FloatTensor([[6.0]]))
print("If u have 6 yrs exp, Salary is:", model(test_exp).data.item())

