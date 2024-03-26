import matplotlib.pyplot as plt
import numpy as np

# 准备存储数据的变量
models_data = {
    'LeNet': {'loss': [], 'accuracy': []},
    'MyNet': {'loss': [], 'accuracy': []},
    'AlexNet': {'loss': [], 'accuracy': []}
}

# 读取每个模型的loss和accuracy数据
model_files = ['LeNet_loss.txt', 'LeNet_acc.txt', 'MyNet_loss.txt', 'MyNet_acc.txt', 'AlexNet_loss.txt', 'AlexNet_acc.txt']
model_labels = ['LeNet', 'LeNet', 'MyNet', 'MyNet', 'AlexNet', 'AlexNet']

for file, label in zip(model_files, model_labels):
    with open(file, 'r') as data_file:
        data = [float(line.strip()) for line in data_file.readlines()]
        if 'loss' in file:
            models_data[label]['loss'] = data
        else:
            models_data[label]['accuracy'] = data

# 准备画布
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 绘制损失图
for model, data in models_data.items():
    ax1.plot(np.arange(1, 21), data['loss'], label=model)

ax1.set_title('Training Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.set_ylim(0, 2)  # 调整纵坐标范围为0-2
ax1.set_yticks(np.arange(0, 2.1, 0.2))  # 设置y轴刻度为0到2
# 绘制准确度图
for model, data in models_data.items():
    ax2.plot(np.arange(1, 21), data['accuracy'], label=model)

ax2.set_title('Training Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
# ax2.set_ylim(70, 100)  # 调整纵坐标范围为0-100
# ax2.set_yticks(np.arange(70, 110, 10))  # 设置y轴刻度为0到100
# 调整布局，以防重叠
plt.tight_layout()

# 显示图形
plt.show()
plt.savefig("./outputs/train/acc_loss.png")
