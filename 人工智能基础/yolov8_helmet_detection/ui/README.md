# 使用说明

本文件夹内主要存放的是基于 pyside6 开发的检测系统前端

## 配置环境

可以根据 requirements.txt 配置环境，如果已经有 pytorch 虚拟环境，可直接补充以下安装包

```Shell
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple  # 安装ultralytics

pip install pyside6 -i https://pypi.tuna.tsinghua.edu.cn/simple  # 安装pyside6

```

## 使用说明

- 直接在配置好的虚拟环境中运行 main.py,运行后将会产生如下界面

![](README_md_files/fab4e170-9e29-11ee-8dc0-2db77a7209e1.jpeg?v=1&type=image)

- 选择模型权重后（在 weights 文件夹中，支持 pt，onnx，engine 格式，目前文件夹中放置了 pt 文件和 onnx 文件）进行模型权重初始化。

- 可以调整 iou 和 confidence 作为检测的阈值

- 可以选择图片、视频和摄像头作为检测对象，步骤都是先选择后检测，然后点击结果展示，可以选择导出结果，按结束键可以关闭当前图像、视频或者摄像头

- 可以读出目标检测框的位置（xyxy 形式）

- 下方的监测框会实时显示操作的过程

## 文件说明

- main_window\.ui 为主界面的 ui 文件，同名的 main_window_ui.py 为 ui 文件编译成的 py 文件。

- weights 文件夹存放权重文件，images 存放了几个测试的图片，video 文件夹存放了一个检测的视频、

- main.py 是主程序，detect.py 是产生检测框的代码

- bg.png 是背景图片
