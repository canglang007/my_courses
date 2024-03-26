from ultralytics import YOLO


model = YOLO('./runs/detect/train/weights/best.pt')

# 验证模型
# 这里batch越大验证越快，默认为1，这里设置为8
# split设置为val指在验证集上检验，如果是test的话就是在测试集上检验
metrics = model.val(split='test',batch=8)  # 无需参数，数据集和设置记忆
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # 包含每个类别的map50-95列表