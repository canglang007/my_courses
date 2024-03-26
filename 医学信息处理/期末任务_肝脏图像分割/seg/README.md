# 医学信息处理期末作业代码说明

## 环境配置

可以按照 requirements.txt 配置环境

## code 部分

包含预处理数据集的 python 文件 data_convertor.py 和 main 文件夹；

main 文件夹中包含了基本的数据增强，读取数据集，训练（train.py 为其他三种网络的训练代码，train_Bi 为 BiSeNetV2 的训练代码（因为损失函数有所不同），和预测的代码，以及网络结构的代码。

## 其他文件说明

本文件夹内包含三个文件夹，除 code 文件夹外，其中 chaos_custom 文件夹中包含经过预处理后的数据集（下载地址链接：<https://pan.baidu.com/s/1pV9CqvVpMkbpzXCK4GRFzg?pwd=tjdx> 提取码：tjdx），results_seg 包含四种网络生成的分割结果（<https://pan.baidu.com/s/1NfyjQg2mvsucZwa2EMvpBg?pwd=tjdx> 提取码：tjdx）以及用于生成评估指标的 evaluate.py 文件。

源数据集下载地址链接：<https://pan.baidu.com/s/12ck6p85TEjgum98MoHq-cg?pwd=tjdx> 提取码：tjdx&#x20;

为保证文件夹大小，图片均以百度网盘链接给出

## 注意事项

详细说明可在实验报告中查看
