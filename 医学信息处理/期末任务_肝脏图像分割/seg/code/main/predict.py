import os
import torch
import logging
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from dataset_pr import HeadSegData
from PIL import Image
from unet.unet_model import UNet

data_path = '../datasets/chaos_custom/test'

net = UNet(3, 2).cuda()
net.load_state_dict(torch.load('./models/200.pth'))

testdata = HeadSegData(data_path, train=False)
test_loader = DataLoader(testdata, batch_size=1, shuffle=False, num_workers=1)
net.eval()

colormap = [
    [0, 0, 0], 
    [255, 255, 255]
]
cm = np.array(colormap).astype('uint8')

dir = "../results_seg/unet/"
if not os.path.exists(dir):
    os.makedirs(dir)
    
if __name__ == '__main__':
    with torch.no_grad():
        for i, (img, img_path) in enumerate(tqdm(test_loader)): # 确保dataloader返回路径
            img = img.cuda(0)
            out = net(img)
            pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
            pre_label = np.asarray(pre_label, dtype=np.uint8)
            
            # 处理图像路径，构建新的文件名
            img_name = os.path.basename(img_path[0]) # 获取原始文件名
            folder_name = os.path.basename(os.path.dirname(img_path[0])) # 获取所在文件夹名
            new_img_name = 'liver_pred_' + img_name.split(',')[0].split('i')[-1] + '.png' # 构建新文件名
            
            # 创建对应文件夹
            save_dir = os.path.join(dir, folder_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            pre = cm[pre_label]
            pre1 = Image.fromarray(pre.astype("uint8"), mode='RGB')
            pre1.save(os.path.join(save_dir, new_img_name)) # 保存在相应的文件夹中