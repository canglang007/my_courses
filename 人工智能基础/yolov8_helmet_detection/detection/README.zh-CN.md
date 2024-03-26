[中文](https://docs.ultralytics.com/zh/) | [한국어](https://docs.ultralytics.com/ko/) | [日本語](https://docs.ultralytics.com/ja/) | [Русский](https://docs.ultralytics.com/ru/) | [Deutsch](https://docs.ultralytics.com/de/) | [Français](https://docs.ultralytics.com/fr/) | [Español](https://docs.ultralytics.com/es/) | [Português](https://docs.ultralytics.com/pt/) | [हिन्दी](https://docs.ultralytics.com/hi/) | [العربية](https://docs.ultralytics.com/ar/)&#x20;

[Ultralytics](https://ultralytics.com) [YOLOv8](https://github.com/ultralytics/ultralytics) 是一款前沿、最先进（SOTA）的模型，基于先前 YOLO 版本的成功，引入了新功能和改进，进一步提升性能和灵活性。YOLOv8 设计快速、准确且易于使用，使其成为各种物体检测与跟踪、实例分割、图像分类和姿态估计任务的绝佳选择。

我们希望这里的资源能帮助您充分利用 YOLOv8。请浏览 YOLOv8 文档 了解详细信息，在 GitHub 上提交问题以获得支持，并加入我们的 Discord 社区进行问题和讨论！

如需申请企业许可，请在 [Ultralytics Licensing](https://ultralytics.com/license) 处填写表格

## 文档

请参阅下面的快速安装和使用示例，以及 [YOLOv8 文档](https://docs.ultralytics.com) 上有关训练、验证、预测和部署的完整文档。

使用 Pip 在一个[Python>\=3.8](https://www.python.org/)环境中安装`ultralytics`包，此环境还需包含[PyTorch>\=1.8](https://pytorch.org/get-started/locally/)。这也会安装所有必要的[依赖项](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt)。

[![PyPI version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

```bash
pip install ultralytics
```

如需使用包括[Conda](https://anaconda.org/conda-forge/ultralytics)、[Docker](https://hub.docker.com/r/ultralytics/ultralytics)和 Git 在内的其他安装方法，请参考[快速入门指南](https://docs.ultralytics.com/quickstart)。

#### CLI

YOLOv8 可以在命令行界面（CLI）中直接使用，只需输入 `yolo` 命令：

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

`yolo` 可用于各种任务和模式，并接受其他参数，例如 `imgsz=640`。查看 YOLOv8 [CLI 文档](https://docs.ultralytics.com/usage/cli)以获取示例。

#### Python

YOLOv8 也可以在 Python 环境中直接使用，并接受与上述 CLI 示例中相同的[参数](https://docs.ultralytics.com/usage/cfg/)：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt")  # 加载预训练模型（建议用于训练）

# 使用模型
model.train(data="coco128.yaml", epochs=3)  # 训练模型
metrics = model.val()  # 在验证集上评估模型性能
results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
```

查看 YOLOv8 [Python 文档](https://docs.ultralytics.com/usage/python)以获取更多示例。

## 模型

在[COCO](https://docs.ultralytics.com/datasets/detect/coco)数据集上预训练的 YOLOv8 [检测](https://docs.ultralytics.com/tasks/detect)，[分割](https://docs.ultralytics.com/tasks/segment)和[姿态](https://docs.ultralytics.com/tasks/pose)模型可以在这里找到，以及在[ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet)数据集上预训练的 YOLOv8 [分类](https://docs.ultralytics.com/tasks/classify)模型。所有的检测，分割和姿态模型都支持[追踪](https://docs.ultralytics.com/modes/track)模式。

所有[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)在首次使用时会自动从最新的 Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)下载。

查看[检测文档](https://docs.ultralytics.com/tasks/detect/)以获取这些在[COCO](https://docs.ultralytics.com/datasets/detect/coco/)上训练的模型的使用示例，其中包括 80 个预训练类别。

| 模型                                                                                 | 尺寸(像素) | mAPval50-95 | 速度 CPU ONNX(ms) | 速度 A100 TensorRT(ms) | 参数(M) | FLOPs(B) |
| ------------------------------------------------------------------------------------ | ---------- | ----------- | ----------------- | ---------------------- | ------- | -------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640        | 37.3        | 80.4              | 0.99                   | 3.2     | 8.7      |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640        | 44.9        | 128.4             | 1.20                   | 11.2    | 28.6     |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640        | 50.2        | 234.7             | 1.83                   | 25.9    | 78.9     |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640        | 52.9        | 375.2             | 2.39                   | 43.7    | 165.2    |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640        | 53.9        | 479.1             | 3.53                   | 68.2    | 257.8    |

- **mAPval** 值是基于单模型单尺度在 [COCO val2017](http://cocodataset.org) 数据集上的结果。 通过 `yolo val detect data=coco.yaml device=0` 复现

- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 COCO val 图像进行平均计算的。 通过 `yolo val detect data=coco.yaml batch=1 device=0|cpu` 复现

查看[检测文档](https://docs.ultralytics.com/tasks/detect/)以获取这些在[Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/)上训练的模型的使用示例，其中包括 600 个预训练类别。

| 模型                                                                                      | 尺寸(像素) | mAP 验证 50-95 | 速度 CPU ONNX(毫秒) | 速度 A100 TensorRT(毫秒) | 参数(M) | 浮点运算(B) |
| ----------------------------------------------------------------------------------------- | ---------- | -------------- | ------------------- | ------------------------ | ------- | ----------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt) | 640        | 18.4           | 142.4               | 1.21                     | 3.5     | 10.5        |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt) | 640        | 27.7           | 183.1               | 1.40                     | 11.4    | 29.7        |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt) | 640        | 33.6           | 408.5               | 2.26                     | 26.2    | 80.6        |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt) | 640        | 34.9           | 596.9               | 2.43                     | 44.1    | 167.4       |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt) | 640        | 36.3           | 860.6               | 3.56                     | 68.7    | 260.6       |

- **mAP 验证** 值适用于在[Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/)数据集上的单模型单尺度。 通过 `yolo val detect data=open-images-v7.yaml device=0` 以复现。

- **速度** 在使用[Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)实例对 Open Image V7 验证图像进行平均测算。 通过 `yolo val detect data=open-images-v7.yaml batch=1 device=0|cpu` 以复现。

查看[分割文档](https://docs.ultralytics.com/tasks/segment/)以获取这些在[COCO-Seg](https://docs.ultralytics.com/datasets/segment/coco/)上训练的模型的使用示例，其中包括 80 个预训练类别。

| 模型                                                                                         | 尺寸(像素) | mAPbox50-95 | mAPmask50-95 | 速度 CPU ONNX(ms) | 速度 A100 TensorRT(ms) | 参数(M) | FLOPs(B) |
| -------------------------------------------------------------------------------------------- | ---------- | ----------- | ------------ | ----------------- | ---------------------- | ------- | -------- |
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640        | 36.7        | 30.5         | 96.1              | 1.21                   | 3.4     | 12.6     |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640        | 44.6        | 36.8         | 155.7             | 1.47                   | 11.8    | 42.6     |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640        | 49.9        | 40.8         | 317.0             | 2.18                   | 27.3    | 110.2    |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640        | 52.3        | 42.6         | 572.4             | 2.79                   | 46.0    | 220.5    |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640        | 53.4        | 43.4         | 712.1             | 4.02                   | 71.8    | 344.1    |

- **mAPval** 值是基于单模型单尺度在 [COCO val2017](http://cocodataset.org) 数据集上的结果。 通过 `yolo val segment data=coco-seg.yaml device=0` 复现

- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 COCO val 图像进行平均计算的。 通过 `yolo val segment data=coco-seg.yaml batch=1 device=0|cpu` 复现

查看[姿态文档](https://docs.ultralytics.com/tasks/pose/)以获取这些在[COCO-Pose](https://docs.ultralytics.com/datasets/pose/coco/)上训练的模型的使用示例，其中包括 1 个预训练类别，即人。

| 模型                                                                                                 | 尺寸(像素) | mAPpose50-95 | mAPpose50 | 速度 CPU ONNX(ms) | 速度 A100 TensorRT(ms) | 参数(M) | FLOPs(B) |
| ---------------------------------------------------------------------------------------------------- | ---------- | ------------ | --------- | ----------------- | ---------------------- | ------- | -------- |
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640        | 50.4         | 80.1      | 131.8             | 1.18                   | 3.3     | 9.2      |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640        | 60.0         | 86.2      | 233.2             | 1.42                   | 11.6    | 30.2     |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640        | 65.0         | 88.8      | 456.3             | 2.00                   | 26.4    | 81.0     |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640        | 67.6         | 90.0      | 784.5             | 2.59                   | 44.4    | 168.6    |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640        | 69.2         | 90.2      | 1607.1            | 3.73                   | 69.4    | 263.2    |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280       | 71.6         | 91.2      | 4088.7            | 10.04                  | 99.1    | 1066.4   |

- **mAPval** 值是基于单模型单尺度在 [COCO Keypoints val2017](http://cocodataset.org) 数据集上的结果。 通过 `yolo val pose data=coco-pose.yaml device=0` 复现

- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 COCO val 图像进行平均计算的。 通过 `yolo val pose data=coco-pose.yaml batch=1 device=0|cpu` 复现

查看[分类文档](https://docs.ultralytics.com/tasks/classify/)以获取这些在[ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/)上训练的模型的使用示例，其中包括 1000 个预训练类别。

| 模型                                                                                         | 尺寸(像素) | acctop1 | acctop5 | 速度 CPU ONNX(ms) | 速度 A100 TensorRT(ms) | 参数(M) | FLOPs(B) at 640 |
| -------------------------------------------------------------------------------------------- | ---------- | ------- | ------- | ----------------- | ---------------------- | ------- | --------------- |
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224        | 66.6    | 87.0    | 12.9              | 0.31                   | 2.7     | 4.3             |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224        | 72.3    | 91.1    | 23.4              | 0.35                   | 6.4     | 13.5            |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224        | 76.4    | 93.2    | 85.4              | 0.62                   | 17.0    | 42.7            |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224        | 78.0    | 94.1    | 163.0             | 0.87                   | 37.5    | 99.7            |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224        | 78.4    | 94.3    | 232.0             | 1.01                   | 57.4    | 154.8           |

- **acc** 值是模型在 [ImageNet](https://www.image-net.org/) 数据集验证集上的准确率。 通过 `yolo val classify data=path/to/ImageNet device=0` 复现

- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 ImageNet val 图像进行平均计算的。 通过 `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu` 复现

## 集成

我们与领先的 AI 平台的关键整合扩展了 Ultralytics 产品的功能，增强了数据集标签化、训练、可视化和模型管理等任务。探索 Ultralytics 如何与[Roboflow](https://roboflow.com/?ref=ultralytics)、ClearML、[Comet](https://bit.ly/yolov8-readme-comet)、Neural Magic 以及[OpenVINO](https://docs.ultralytics.com/integrations/openvino)合作，优化您的 AI 工作流程。

|                                                 Roboflow                                                  |                                            ClearML ⭐ NEW                                            |                                                     Comet ⭐ NEW                                                     |                                        Neural Magic ⭐ NEW                                        |
| :-------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------: |
| 使用 [Roboflow](https://roboflow.com/?ref=ultralytics) 将您的自定义数据集直接标记并导出至 YOLOv8 进行训练 | 使用 [ClearML](https://cutt.ly/yolov5-readme-clearml)（开源！）自动跟踪、可视化，甚至远程训练 YOLOv8 | 免费且永久，[Comet](https://bit.ly/yolov8-readme-comet) 让您保存 YOLOv8 模型、恢复训练，并以交互式方式查看和调试预测 | 使用 [Neural Magic DeepSparse](https://bit.ly/yolov5-neuralmagic) 使 YOLOv8 推理速度提高多达 6 倍 |

## Ultralytics HUB

体验 [Ultralytics HUB](https://bit.ly/ultralytics_hub) ⭐ 带来的无缝 AI，这是一个一体化解决方案，用于数据可视化、YOLOv5 和即将推出的 YOLOv8 🚀 模型训练和部署，无需任何编码。通过我们先进的平台和用户友好的 [Ultralytics 应用程序](https://ultralytics.com/app_install)，轻松将图像转化为可操作的见解，并实现您的 AI 愿景。现在就开始您的**免费**之旅！

## 贡献

我们喜欢您的参与！没有社区的帮助，YOLOv5 和 YOLOv8 将无法实现。请参阅我们的[贡献指南](https://docs.ultralytics.com/help/contributing)以开始使用，并填写我们的[调查问卷](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)向我们提供您的使用体验反馈。感谢所有贡献者的支持！🙏

## 许可证

Ultralytics 提供两种许可证选项以适应各种使用场景：

- **AGPL-3.0 许可证**：这个[OSI 批准](https://opensource.org/licenses/)的开源许可证非常适合学生和爱好者，可以推动开放的协作和知识分享。请查看[LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 文件以了解更多细节。

- **企业许可证**：专为商业用途设计，该许可证允许将 Ultralytics 的软件和 AI 模型无缝集成到商业产品和服务中，从而绕过 AGPL-3.0 的开源要求。如果您的场景涉及将我们的解决方案嵌入到商业产品中，请通过 [Ultralytics Licensing](https://ultralytics.com/license)与我们联系。

## 联系方式

对于 Ultralytics 的错误报告和功能请求，请访问 [GitHub Issues](https://github.com/ultralytics/ultralytics/issues)，并加入我们的 [Discord](https://ultralytics.com/discord) 社区进行问题和讨论！
