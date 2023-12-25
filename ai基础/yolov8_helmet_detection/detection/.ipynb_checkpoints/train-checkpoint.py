from ultralytics import YOLO
import torch
from torch.utils.tensorboard import SummaryWriter


model = YOLO('./weight/yolov8n.pt') # 加载预训练模型


if __name__ == '__main__':
    results = model.train(data='./helm.yaml', epochs=100, patience=30,save_period=40, device=[0,1],optimizer='SGD') 



