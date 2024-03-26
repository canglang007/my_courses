
from torch import nn
import torch.nn.functional as F
import torch
from torch.nn import ReLU, Conv2d, MaxPool2d, AdaptiveAvgPool2d, \
                     Flatten, Linear, Dropout
import random


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # 输入x，经过第一个卷积层，使用relu激活函数，最大池化窗口大小为2
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 输入x，经过第二个卷积层，使用relu激活函数，最大池化窗口大小为2，经过dropout，比例为0.5
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 将x的维度转换为(-1, 320)
        x = x.view(-1, 320)
        # 输入x，经过第一个全连接层，使用relu激活函数
        x = F.relu(self.fc1(x))
        # 对x进行dropout，比例为0.5，训练模式为self.training
        x = F.dropout(x, training=self.training)
        # 输入x，经过第二个全连接层，输出维度为10
        x = self.fc2(x)
        # 返回x的log_softmax
        return F.log_softmax(x, dim=1)
    
    # 构建网络
class LeNet(nn.Module): 					    
    def __init__(self):						
        super(LeNet, self).__init__()    	
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),  # 输出为6*28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出为6*14*14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 输出为16*10*10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出为16*5*5
        )
        self.block_2 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):  # 正向传播过程
        x = self.block_1(x)
        x = x.view(-1,16*5*5)
        x = self.block_2(x)
        return x



class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # 调整输出尺寸
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 调整输入尺寸
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # model = AlexNet()
    model = LeNet()
    # model = MyNet()
    # model = VGG16()
    print(model)
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)

