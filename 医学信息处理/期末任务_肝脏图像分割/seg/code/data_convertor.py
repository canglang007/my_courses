import os
import random
import pydicom
from PIL import Image
import numpy as np

# 设定随机种子以确保结果的可重复性
random.seed(42)

# 指定数据集的路径
data_dir = './CT'  # 当前文件夹路径
output_dir = './CT_trainval/'
train_ratio = 0.7

# 读取所有的数据集的文件夹，isdir是判断是不是文件夹，然后进行排序
folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
folders = sorted(folders, key=lambda x: int(x))
random.shuffle(folders)
# 训练集和验证集的大小
train_size = int(len(folders) * train_ratio)
train_folders = folders[:train_size]
test_folders = folders[train_size:]

# 将文件夹中的dcm文件转变为png文件
def convert_dicom_to_png(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    dicom_files = [f for f in os.listdir(input_folder) if f.endswith('.dcm')]

    for file in dicom_files:
        dicom_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file.replace('.dcm', '.png'))
        # 使用dcmread来读取dcm文件
        ds = pydicom.dcmread(dicom_path)
        # 获取 DICOM 文件中的像素数据
        pixel_array = ds.pixel_array

        # 中位数和四分位数排除异常
        q25, q75 = np.percentile(pixel_array, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        # 将数组中的元素限定在范围内
        pixel_array = np.clip(pixel_array, lower_bound, upper_bound)

        min_value = np.min(pixel_array)
        max_value = np.max(pixel_array)
        # 进行归一化操作，统一到0-255范围内
        pixel_array = (pixel_array - min_value) / (max_value - min_value) * 255
        pixel_array = pixel_array.astype(np.uint8)
        # 将图像转化为灰度图片
        image = Image.fromarray(pixel_array).convert('L')
        image.save(output_path)

# 复制ground_truth
def copy_ground_truth(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    for file in png_files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        img = Image.open(input_path)
        img.save(output_path)

# 处理训练集和测试集
for mode, folders in [('train', train_folders), ('test', test_folders)]:
    for folder in folders:
        dicom_folder = os.path.join(data_dir, folder, 'DICOM_anon')
        ground_folder = os.path.join(data_dir, folder, 'Ground')
        output_dicom_folder = os.path.join(output_dir, mode, 'DICOM', folder)
        output_ground_folder = os.path.join(output_dir, mode, 'Ground', folder)
        
        # 转换DICOM为PNG并保存到指定位置
        convert_dicom_to_png(dicom_folder, output_dicom_folder)
        # 复制Ground真值到指定位置
        copy_ground_truth(ground_folder, output_ground_folder)
