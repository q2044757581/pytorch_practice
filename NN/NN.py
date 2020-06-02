import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as opt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.h1 = nn.Linear(2, 2)
        self.h2 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.sigmoid(self.h1(x))
        x = F.sigmoid(self.h2(x))
        return x


net = Net()
print(net)
params = list(net.parameters())
print(len(params))
X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
X = torch.from_numpy(X).float()
Y = np.array([1, 0, 0, 1])
Y = torch.from_numpy(Y)
Y = Y.view(-1, 1).float()
optimizer = opt.Adam(net.parameters(), lr=0.001)
for i in range(10000):
    # 清除梯度
    optimizer.zero_grad()
    # 计算输出
    y_hat = net(X)
    # 计算loss
    loss = F.mse_loss(y_hat, Y)
    print(i, loss)
    # loss反向传播
    loss.backward()
    optimizer.step()

print(net(X))
