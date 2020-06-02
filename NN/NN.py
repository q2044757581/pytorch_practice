import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.h1 = nn.Linear(2, 2)
        self.h2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.h1(x)
        x = self.h2(x)
        return F.sigmoid(x)


X = torch.Tensor()
