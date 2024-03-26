import os
import logging
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import *
from tqdm import tqdm
import click
import torch.nn.functional as F
import numpy as np
from unet.unet_model import UNet
from BiSeNetV2 import BiSeNetV2
import os
import time 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


@click.command()
@click.option('--data-path', type=str, help='Path to dataset folder', default='../datasets/chaos_custom/train')
@click.option('--models-path', type=str, help='Path for storing model snapshots', default='./models')
@click.option('--backend', type=str, default='resnet34', help='Feature extractor')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--crop_x', type=int, default=1000, help='Horizontal random crop size')
@click.option('--crop_y', type=int, default=304, help='Vertical random crop size')
@click.option('--batch-size', type=int, default=32)
@click.option('--alpha', type=float, default=0.4, help='Coefficient for classification loss term')
@click.option('--epochs', type=int, default=50, help='Number of training epochs to run')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--start-lr', type=float, default=0.1)
@click.option('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')

# 定义训练函数，用于训练模型
def train(data_path, models_path, backend, snapshot, crop_x, crop_y, batch_size, alpha, epochs, start_lr, milestones, gpu):
    # 创建models_path文件夹，如果已存在，则不创建
    os.makedirs(models_path, exist_ok=True)
    # 创建HeadSegData实例，用于加载数据
    traindata = HeadSegData(data_path)
    # 创建DataLoader实例，用于加载数据
    train_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=64)

    # 创建UNet实例，用于构建模型
    

    net = BiSeNetV2(num_classes=3).cuda(0)
    # 创建交叉熵损失函数实例，用于计算损失
    
    seg_criterion = nn.CrossEntropyLoss().cuda(0)
    # 创建Adam优化器实例，用于优化模型参数
    optimizer = optim.SGD(net.parameters(), lr=start_lr)

    print("start training...")
    start_time = time.time()
    # 开始训练
    for epoch in range(epochs):
        # 设置模型为训练模式
        net.train()
        # 如果epoch为6的倍数且不为0，则将学习率乘以0.5
        if epoch % 6 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5
        # 遍历训练数据
        for i, (x, y) in enumerate(train_loader):
            # 将训练数据转换为cuda格式
            x, y = x.cuda(0), y.cuda(0).long()
            # 运行模型，计算输出
            out = net(x)
            # 计算损失
            seg_loss =seg_criterion(out[0],y)
            aux_loss_1 = seg_criterion(out[1], y)
            aux_loss_2 = seg_criterion(out[2], y)
            aux_loss_3 = seg_criterion(out[3], y)
            aux_loss_4 = seg_criterion(out[4], y)

 
            loss =seg_loss + 0.2*aux_loss_1 + 0.2*aux_loss_2 + 0.2*aux_loss_3 + 0.2*aux_loss_4
            # 每50次迭代输出一次损失
            if i % 50 == 0:
                status = '[batch:{0}/{1} epoch:{2}] loss = {3:0.5f}'.format(i, len(traindata)//batch_size, epoch + 1, loss.item())
                print(status)
            
            # 梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

        # 每训练完6个epoch，保存一次模型
        torch.save(net.state_dict(), os.path.join(models_path, str(epoch+1)+".pth"))
        end_time = time.time()
        # 计算训练时间
        elapsed_time = end_time - start_time
        # 打印训练时间
        print(f'Training time for epoch {epoch + 1}: {elapsed_time} seconds')
        
if __name__ == '__main__':
    # data_path = "../chaos_custom/train/"
    #os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    print(torch.cuda.is_available())
    train()
    # data_path=data_path,models_path="./checkpoints",batch_size=32,epochs=200,start_lr=0.1