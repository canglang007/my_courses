import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.preprocessing import label_binarize
from model.network import *  # 导入你的模型定义
import os
import numpy as np

def evaluate_model(model, test_loader, criterion, class_names, model_name):
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(nn.functional.softmax(outputs, dim=1).cpu().numpy())

    test_loss = criterion(outputs, labels).item()
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    conf_matrix = confusion_matrix(all_labels, all_predictions)
    plot_confusion_matrix(conf_matrix, class_names, model_name)

    # 计算多类别的AUROC
    y_true_bin = label_binarize(all_labels, classes=list(range(len(class_names))))
    y_score = np.array(all_probs)
    plot_roc_curve_multiclass(y_true_bin, y_score, class_names, model_name)

    return test_loss, accuracy, precision, recall, f1, all_predictions, all_labels

def plot_roc_curve_multiclass(y_true, y_score, class_names, model_name):
    n_classes = len(class_names)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve for class {class_names[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve for Multiclass')
    plt.legend(loc='lower right')
    
    save_dir = './outputs/test/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, '{}_roc_multiclass.png'.format(model_name)))
    plt.show()


def plot_confusion_matrix(conf_matrix, class_names,model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    plt.savefig('./outputs/test/{}_matrix.png'.format(model_name))
# 定义超参数
batch_size = 64

# 数据预处理

# 加载测试数据集
test_data = torchvision.datasets.FashionMNIST(root='./data/fashionMinst', train=False, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                        ]))

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# 定义类别标签
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 初始化模型
LeNet_model = LeNet()
AlexNet_model = AlexNet()
MyNet_model = MyNet()

# 加载模型权重
LeNet_model.load_state_dict(torch.load('./points/LeNet.pth'))
AlexNet_model.load_state_dict(torch.load('./points/AlexNet.pth'))
MyNet_model.load_state_dict(torch.load('./points/MyNet.pth'))

# 评估模型
LeNet_results = evaluate_model(LeNet_model, test_loader, nn.CrossEntropyLoss(), class_names, 'LeNet')
AlexNet_results = evaluate_model(AlexNet_model, test_loader, nn.CrossEntropyLoss(), class_names, 'AlexNet')
MyNet_results = evaluate_model(MyNet_model, test_loader, nn.CrossEntropyLoss(), class_names, 'MyNet')

