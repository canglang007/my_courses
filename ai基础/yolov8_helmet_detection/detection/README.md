# 主要代码说明

## 数据集说明

本数据集来源是 kaggle 的一个头盔检测数据集，来源是[YOLO helmet/head (kaggle.com)](https://www.kaggle.com/datasets/vodan37/yolo-helmethead)。

由于下载可能失败，在这里提供自己下载下来的链接：<https://pan.baidu.com/s/1ke_YZZFZIDeAqkzkDrNPFA?pwd=tjdx>
提取码：tjdx
\--来自百度网盘超级会员 V2 的分享

为保证文件大小，本文件夹不放置数据集

如果要使用，下载后请改名为 datasets

## 环境配置

本程序是基于 yolov8 进行的训练，所以需要先配置 yolo 环境，这里采用的是从 github clone 的方法，在 git bash 中运行下面的指令

```Shell
git clone git@github.com:ultralytics/ultralytics.git
```

下载后进入该文件夹中，根据里面的 requirements.txt 配置环境：

    pip install -r requirements.txt

## 代码说明

在官方的 ultralytics 文件夹基础上，主要加入了 train.py，test.py, predict.py, pth2onnx.py,数据集 dataset 文件夹，配置文件 helm.yaml。

## 使用说明

- 首先准备好数据集后改变数据集最外层目录为 datasets（里层是 images 和 labels),编写数据集调用的配置文件，放在合适的位置（与 train.py 同目录)

- 由于本实验以 yolov8 官方的预训练模型为基础进行训练，所以在训练前需要提前下载（自动下载可能由于网络不稳定不易下载）权重文件到 weights 中（采用的是 yolov8n.pth）,下载链接在官方的 readme 文档中直接点击可以下载。

- 训练直接运行 train.py 即可，里面可以修改训练的参数，本次实验的训练参数如下：

  ```python
  results = model.train(data='./helm.yaml', epochs=100, patience=30,save_period=40, device=[0,1],optimizer='SGD')
  ```

- 训练完成后会自动在当前目录下生成一个 runs 文件夹，里面有 detect 文件夹，然后里面是训练过程的记录

  在 runs/detect/train/weights 中包含训练产生的权重文件（只保留了 best 与 last 权重，last 权重可支持断点重训），runs/detect/train 中记录了其他训练过程的评估指标（包括训练集上和验证集上的指标）以及 tensorboard 文件

- 训练完成后可以进行测试（验证），运行 test.py 即可，评估的指标将会产生在 runs/detect/val 中

- 训练完还可以进行预测（推理),运行 predict.py 即可，预测对象可以是图片，视频，网页视频等，在这里改动（修改 source 为合适的路径即可）：

  ```python
  model.predict(source='./datasets/images/test', save=True, conf=0.25)
  ```

&#x20; 选择 save\=True 后将会把预测结果保存在 runs/detect/predict 中，为保证文件大小，这里仅仅保留在测试集上的前五张图片。

- 使用 pth2onnx.py 可以把 pth 权重文件转为 onnx 权重文件，这样在进行模型部署的时候推理速度更快
