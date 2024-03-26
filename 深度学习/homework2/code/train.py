import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import os

from model import RNN,LSTM,GRU
from dataloader import get_dataloader

def get_embedding_vectors(model):
    embedding_layer = model.embedding 
    embedding_vectors = embedding_layer.weight.data.cpu().numpy()
    return embedding_vectors

def get_current_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# 计算acc
def calculate_accuracy(model, data_loader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 计算验证损失
def calculate_validation_loss(model, validation_loader, criterion, device='cuda'):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(validation_loader)


def train(model, train_loader, validation_loader, vocab_list, num_epochs=50, learning_rate=0.005, device='cuda',model_name='RNN'):
    train_acc_lst, val_acc_lst = [], []
    train_loss_lst, val_loss_lst = [], []
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9,weight_decay=0.0005)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = f'logs/{current_time}'
    writer = SummaryWriter(log_dir=log_dir)

    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, f'{model_name}_best_model.pth')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')):
            inputs, labels = inputs.to(device), labels.to(device)
            # 梯度清零
            optimizer.zero_grad()

            outputs = model(inputs)
            # 计算损失函数
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar('training_loss', loss.item(), epoch)
        # 计算平均的训练损失
        average_train_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_train_loss:.4f}")
  
        # Validation Loss Calculation
        val_loss = calculate_validation_loss(model, validation_loader, criterion) 

        # 记录验证集上的损失
        writer.add_scalar('validation_loss', val_loss, epoch)

        # 记录训练集和验证集上的acc
        train_acc = calculate_accuracy(model, train_loader)
        writer.add_scalar('train_acc',train_acc, epoch)
        accuracy = calculate_accuracy(model, validation_loader) 
        writer.add_scalar('validation_accuracy', accuracy, epoch)
        
        train_loss_lst.append(average_train_loss)
        train_acc_lst.append(train_acc)
        val_loss_lst.append(val_loss)
        val_acc_lst.append(accuracy)
        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param, epoch)

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         writer.add_scalar(f'{name}_grad_norm', param.grad.norm().item(), epoch)

        # 记录学习率变化
        current_lr = get_current_learning_rate(optimizer)  # Implement a function to fetch current LR
        writer.add_scalar('learning_rate', current_lr, epoch)

        # Save best model
        if average_train_loss < best_val_loss:
            best_val_loss = average_train_loss
            # 保存模型
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with loss: {best_val_loss:.4f}")

    writer.close()
    file_1 = '{}_train_acc.txt'.format(model_name)
    file_2 = '{}_train_loss.txt'.format(model_name)
    file_3 = '{}_val_acc.txt'.format(model_name)
    file_4 = '{}_val_loss.txt'.format(model_name)
    
    with open(file_1, 'w') as file:
        for num in train_acc_lst:
            file.write("%s\n" % str(num))

    with open(file_2, 'w') as file:
        for num in train_loss_lst:
            file.write("%s\n" % str(num))
    with open(file_3, 'w') as file:
        for num in val_acc_lst:
            file.write("%s\n" % str(num))

    with open(file_4, 'w') as file:
        for num in val_loss_lst:
            file.write("%s\n" % str(num))


# 部分超参数与路径设置
#------------------------------------------------
# 数据集路径
data_dir = './aclImdb'

batch_size = 32
num_steps = 500
val_split = 0.2
# model_name = "RNN"

# 读取dataloader
train_loader, validation_loader, _, vocab_size, vocab_list = get_dataloader(data_dir, batch_size, num_steps, val_split)
print(f'vocab_size:{vocab_size}')
# if model_name == 'RNN':
#     model = RNN(vocab_size=vocab_size, embedding_dim=512, hidden_size=128)
# elif model_name == 'LSTM':
#     model = LSTM(vocab_size=vocab_size, embedding_dim=512, hidden_size=128)
# elif model_name == 'GRU':
#     model = RNN(vocab_size=vocab_size, embedding_dim=512, hidden_size=128)     
print("开始训练RNN网络")
model = RNN(vocab_size=vocab_size, embedding_dim=512, hidden_size=128)
train(model, train_loader, validation_loader, vocab_list, num_epochs=50, device='cuda',model_name='RNN')
print("RNN训练完毕")
print("开始训练LSTM网络")
model = LSTM(vocab_size=vocab_size, embedding_dim=512, hidden_size=128)
train(model, train_loader, validation_loader, vocab_list, num_epochs=50, device='cuda',model_name='LSTM')
print("LSTM训练完毕")
print("开始训练GRU网络")
model = RNN(vocab_size=vocab_size, embedding_dim=512, hidden_size=128) 
train(model, train_loader, validation_loader, vocab_list, num_epochs=50, device='cuda',model_name='GRU')
print("GRU网络训练完毕")