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
from pspnet import PSPNet
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"


@click.command()
@click.option('--data-path', type=str, help='Path to dataset folder', default='../datasets/chaos_custom/train')
@click.option('--models-path', type=str, help='Path for storing model snapshots', default='./models')
@click.option('--backend', type=str, default='resnet34', help='Feature extractor')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--crop_x', type=int, default=1000, help='Horizontal random crop size')
@click.option('--crop_y', type=int, default=304, help='Vertical random crop size')
@click.option('--batch-size', type=int, default=32)
@click.option('--alpha', type=float, default=0.4, help='Coefficient for classification loss term')
@click.option('--epochs', type=int, default=200, help='Number of training epochs to run')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--start-lr', type=float, default=0.1)
@click.option('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')

def train(data_path, models_path, backend, snapshot, crop_x, crop_y, batch_size, alpha, epochs, start_lr, milestones, gpu):
    os.makedirs(models_path, exist_ok=True)
    # 读取数据
    traindata = HeadSegData(data_path)
    # 加载dataloader
    train_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=64)

    net = UNet(3, 2).cuda(0)
    net = PSPNet(2)
    # 损失函数
    seg_criterion = nn.CrossEntropyLoss().cuda(0)
    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=start_lr)

    print("start training...")
    
    for epoch in range(epochs):
        net.train()
        # epoch每满6的倍数
        if epoch % 6 == 0 and epoch != 0:
            # 调整学习率
            for group in optimizer.param_groups:
                group['lr'] *= 0.5

        for i, (x, y) in enumerate(train_loader):
            x, y = x.cuda(0), y.cuda(0).long()
            # 计算输出
            out = net(x)
            # 计算输出
            loss = seg_criterion(out, y)

            if i % 50 == 0:
                status = '[batch:{0}/{1} epoch:{2}] loss = {3:0.5f}'.format(i, len(traindata)//batch_size, epoch + 1, loss.item())
                print(status)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), os.path.join(models_path, str(epoch+1)+".pth"))
        
if __name__ == '__main__':
    train()
