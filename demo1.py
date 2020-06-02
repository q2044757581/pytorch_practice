import torch
import numpy as np
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

# a = torch.empty(3, 3)
# a = torch.ones(3, 3)
# a = torch.rand(3, 3, 3)
# a = torch.zeros(3, 3)

a = torch.ones(2, 2)
b = torch.ones(2, 2)
# .numpy() 可以转 numpy array
print(torch.matmul(a, b).numpy())
np_a = np.array([2, 2, 2])
print(torch.from_numpy(np_a))
c = b.view(1, 4)
print(c)
# 加法
# print(a + b)
# print(torch.add(a, b))
# result = torch.empty(2, 2)
# torch.add(a, b, out=result)
# print(result)