
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from model.network import *
import argparse
from torch.optim import lr_scheduler
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
import os

def train_model(model, device, train_loader, epoch,model_name):
    model = model.to(device)
    train_acc_lst, test_acc_lst = [], []
    train_loss_lst, tset_loss_lst = [], []

    # 记录训练过程中最大的精度
    max_train_acc = 0
    max_test_acc = 0
    start_time = time.time()
     # 定义优化器（SGD：随机梯度下降）
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # 学习率每隔 10 个 epoch 变为原来的 0.1
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    loss_fn = nn.CrossEntropyLoss()
    for i in range(epoch):
        print("---------开始第{}/{}轮训练，本轮学习率为：{}---------".format((i + 1), epoch, scheduler.get_last_lr()))
        # 记录每轮训练批次数，每100次进行一次输出
        count_train = 0
        
        # 训练步骤开始
        model.train() # 将网络设置为训练模式，当网络包含 Dropout, BatchNorm时必须设置，其他时候无所谓
        for (features, targets) in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            # 梯度清零，也就是把loss关于weight的导数变成0.
           
            optimizer.zero_grad()
            
            # 获取网络输出
            output = model(features)
            
            # 获取损失
            loss = loss_fn(output, targets)
            
            # 反向传播
            loss.backward()
            # 训练
            optimizer.step()
            # 纪录训练次数
            count_train += 1
            # item()函数会直接输出值，比如tensor(5),会输出5
            if count_train % 100 == 0:
                # 记录时间
                end_time = time.time()
                print(f"训练批次{count_train}/{len(train_loader)}, loss: {loss.item():.3f}，用时：{(end_time - start_time):.2f}" )

        # 将网络设置为测试模式，当网络包含 Dropout, BatchNorm时必须设置，其他时候无所谓
        model.eval()
        with torch.no_grad():
            # 计算训练精度
            train_accuracy, train_loss = compute_accuracy_and_loss(model, train_data, train_loader, device=device)
            # 更新最高精度
            if train_accuracy > max_train_acc:
                max_train_acc = train_accuracy
            
            # 计算测试精度
            test_accuracy, test_loss = compute_accuracy_and_loss(model, test_data, test_loader, device=device)
            # 更新最高精度
            if test_accuracy > max_test_acc:
                max_test_acc = test_accuracy
            
            # 收集训练过程精度和loss
            train_loss_lst.append(train_loss)
            train_acc_lst.append(train_accuracy)
            tset_loss_lst.append(test_loss)
            test_acc_lst.append(test_accuracy)
            
            print(f'Train Loss.: {train_loss:.2f}' f' | Validation Loss.: {test_loss:.2f}')
            print(f'Train Acc.: {train_accuracy:.2f}%' f' | Validation Acc.: {test_accuracy:.2f}%')

        # 训练计时
        elapsed = (time.time() - start_time) / 60
        print(f'本轮训练累计用时: {elapsed:.2f} min')

        # 保存达标的训练的模型
        if test_accuracy > 70:
            torch.save(model.state_dict(), "./points/{}.pth".format(model_name))
            print("第{}次训练模型已保存".format(i + 1))
        
        # 更新学习率
        scheduler.step()
    file_1 = '{}_acc.txt'.format(model_name)
    file_2 = '{}_loss.txt'.format(model_name)
    with open(file_1, 'w') as file:
        for num in train_acc_lst:
            file.write("%s\n" % str(num))

    with open(file_2, 'w') as file:
        for num in train_loss_lst:
            file.write("%s\n" % str(num))
    return train_loss_lst, train_acc_lst


# 计算精度和损失的函数
def compute_accuracy_and_loss(model, dataset, data_loader, device):
    correct, total = .0, .0
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)
        output = model(features)
        currnet_loss = loss_fn(output, targets)
        # 求预测结果精确度之和
        # argmax:求最大值的下标，1按行求，0按列求
#         correct += (output.argmax(1) == targets).sum()
        
        _, predicted_labels = torch.max(output, 1)
        correct += (predicted_labels == targets).sum()

        
        total += targets.size(0)
        
    return float(correct) * 100 / len(dataset), currnet_loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=20, help="epoch")
    parser.add_argument("--batch_size_train", type=int, default=64,help="batch_size_train")
    parser.add_argument("--batch_size_test", type=int, default=1000, help="batch_size_test")
    parser.add_argument("--momentum", default=0.5, help="Adjusting optimizer momentum parameters")
    parser.add_argument("--log_interval", type=int, default=10, help="log_interval")
    parser.add_argument("--random_seed", type=int, default=1, help="random_seed")
    parser.add_argument("--model", type=str, default='LeNet',choices=['MyNet', 'AlexNet', 'LeNet', 'VGG16'], help="network_name(MyNet/LeNet/AlexNet/VGG16)")   
    parser.add_argument("--lr", default=0.01, help="learning_rate")
    parser.add_argument("--data_root",type=str, default='./data/fashionMinst', help="dataset_root")
    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    # 加载数据
    train_data = torchvision.datasets.FashionMNIST(root=args.data_root, train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    (0.1307,), (0.3081,))
                                            ]))

    test_data = torchvision.datasets.FashionMNIST(root=args.data_root, train=False, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                        ]))
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("训练集的长度为{}".format(train_data_size))
   
    # 加载数据集
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    epoch = args.epoch
    
    loss_fn = nn.CrossEntropyLoss()

    #-----------------------------------------------------
    print("开始训练LeNet网络")
    LeNet_loss_lst, LeNet_acc_lst = train_model(LeNet(), device,train_loader,epoch,'LeNet')
    print("LeNet训练完毕")
    print("开始训练AlexNet网络")
    AlexNet_loss_lst, AlexNet_acc_lst = train_model(AlexNet(), device,train_loader,epoch, 'AlexNet')
    print("AlexNet训练完毕")
    print("开始训练MyNet网络")
    MyNet_loss_lst, MyNet_acc_lst = train_model(MyNet(), device,train_loader,epoch,'MyNet')
    print("MyNet训练完毕")
 


    

