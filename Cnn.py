import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # (2, 2)大小的最大池化层
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果大小是正方形，则只能指定一个数字
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        print(x.size())      # torch.Size([1, 16, 6, 6])
        size = x.size()[1:]  # 除batch维度外的所有维度
        print(size)          # torch.Size([16, 6, 6])
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)  # 打印模型结构