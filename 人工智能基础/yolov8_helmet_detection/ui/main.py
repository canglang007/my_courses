import random
import sys
import numpy as np
from PySide6 import QtWidgets
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog
from PySide6 import QtGui
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QGuiApplication

from main_window_ui import Ui_MainWindow
from detect import draw_detections

import sys
import shutil
from pathlib import Path

import cv2
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  

# 将图片转换成QT格式的图片
def convert2QImage(img):
    height, width, channel = img.shape
    return QImage(img, width, height, width * channel, QImage.Format_RGB888)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        screen = QGuiApplication.primaryScreen()
        center = screen.geometry().center()
        x_pos = center.x() - self.width() // 2
        y_pos = center.y() - self.height() // 2
        self.move(x_pos,y_pos)

        self.input_width = self.input.width()
        self.input_height = self.input.height()
        # self.output_width = self.output.width()
        # self.output_height = self.output.height()
        self.imgsz = 640
        self.timer = QTimer()
        self.timer.setInterval(1)
        self.timer_c = QTimer(self)
        self.timer_c.timeout.connect(self.detect_camera)
        self.video = None
        self.out = None

        # 使用gpu，如果只有cpu要改一下
        self.device = "cuda:0"
        self.num_stop = 1
        self.numcon = self.con_slider.value() / 100.0
        self.numiou = self.iou_slider.value() / 100.0
        self.results = []
        self.camera = None
        self.running = False
        self.bind_slots()
        self.init_icons()

    # 打开图片
   # 打开图片函数
    def open_image(self):
        # 停止计时器
        self.timer.stop()
        # 设置文件对话框的选项
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        # 获取文件路径
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "./", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)", options=options)
        # 如果文件路径存在
        if self.file_path:
            # 创建文件对话框
            dialog = QFileDialog(self, "Open File", self.file_path)
            # 设置文件对话框的尺寸
            dialog.resize(800, 600)
            # 关闭文件对话框
            dialog.close()
            # 加载图片
            pixmap = QPixmap(self.file_path)
            # 缩放图片，使其宽高比保持不变
            scaled_pixmap = pixmap.scaled(640, 480, aspectMode=Qt.KeepAspectRatio)
            # 设置输入框的图片
            self.input.setPixmap(QPixmap(self.file_path))
            # 设置输入框的文本
            self.lineEdit.setText('图片打开成功！！！')
    # 打开视频
    def open_video(self):
        # 停止计时器
        self.timer.stop()
        # 设置文件对话框的选项
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        # 获取视频路径
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Select vidos", dir='./', filter="Videos (*.mp4 *.avi *.gif *.MPEG)", options=options)
        # 如果视频路径存在
        if self.video_path:
            # 创建文件对话框
            dialog = QFileDialog(self, "Open File", self.video_path)
            # 设置文件对话框的大小
            dialog.resize(800, 600)
            # 关闭文件对话框
            dialog.close()
            # 设置视频路径
            self.video_path = self.video_path
            # 读取视频
            self.video = cv2.VideoCapture(self.video_path)


            # 读取视频帧
            ret, frame = self.video.read()
            # 如果读取成功
            if ret:
                # 设置提示信息
                self.lineEdit.setText("成功打开视频！！！")
                # 将视频帧从BGR转换为RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 设置输入帧的宽和高
                dst_size = (self.input_width, self.input_height)
                # 将帧缩放到指定大小
                resized_frame = cv2.resize(frame, dst_size, interpolation=cv2.INTER_AREA)
                # 将帧设置为输入框的图片
                self.input.setPixmap(QPixmap(convert2QImage(resized_frame)))
            # 如果读取失败
            else:
                # 设置提示信息
                self.lineEdit.setText("视频有误，请重新打开！！！")
            # 创建输出视频流
            self.out = cv2.VideoWriter('prediction.mp4', cv2.VideoWriter_fourcc(
                    *'mp4v'), 30, (int(self.video.get(3)), int(self.video.get(4))))
    # 加载模型函数
    def load_model(self):
        # 设置文件对话框的选项
        options = QFileDialog.Options()
        # 设置文件对话框不使用本地对话框
        options |= QFileDialog.DontUseNativeDialog
        # 获取打开的文件名
        self.openfile_name_model, _ = QFileDialog.getOpenFileName(self.button_weight_select, '选择权重文件',
                                                                  'weights/', "Weights (*.pt *.onnx *.engine)", options=options)
        # 如果没有打开的文件名，则弹出警告框
        if not self.openfile_name_model:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"权重打开失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            # 创建文件对话框
            dialog = QFileDialog(self, "Open File", self.openfile_name_model)
            # 设置文件对话框的宽高
            dialog.resize(800, 600)
            # 关闭文件对话框
            dialog.close()
            # 设置提示信息
            result_str = '成功加载模型权重, 权重地址: ' + str(self.openfile_name_model)
            self.lineEdit.setText(result_str)
    # 初始化模型
    def init_model(self):
        from ultralytics import YOLO

        self.weights_path = str(self.openfile_name_model)
        
        self.model = YOLO(self.weights_path)
        self.names = self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]
        print("model initial done")

        QtWidgets.QMessageBox.information(self, u"!", u"模型初始化成功", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        self.lineEdit.setText("成功初始化模型!!!")
    # 开始检测
    def detect_begin(self):
        # name_list = []
        self.img = cv2.imread(self.file_path)
        self.pred = self.model.predict(source=self.img, iou=self.numiou, conf=self.numcon)  # save plotted images
        preprocess_speed = self.pred[0].speed['preprocess']
        inference_speed = self.pred[0].speed['inference']
        postprocess_speed = self.pred[0].speed['postprocess']
        self.lineEdit_detect_time.setText(str(round((preprocess_speed + inference_speed + postprocess_speed) / 1000, 3)))
        self.lineEdit_detect_object_nums.setText(str(self.pred[0].boxes.conf.shape[0]))

        self.results = self.pred[0].boxes.xyxy.tolist()

        if self.pred[0].boxes.conf.shape[0]:
            for i in range(self.pred[0].boxes.conf.shape[0]):
                self.comboBox.addItem('目标' + str(i + 1))
        
        QtWidgets.QMessageBox.information(self, u"!", u"成功检测图像", buttons=QtWidgets.QMessageBox.Ok,
                                      defaultButton=QtWidgets.QMessageBox.Ok)
    
        self.lineEdit.setText("成功检测图像!!!") 

    def convert2QImage(img):
        height, width, channel = img.shape
        return QImage(img, width, height, width * channel, QImage.Format_RGB888)

    def detect_show(self):
        conf_list = self.pred[0].boxes.conf.tolist()
        cls_list_int = [int(i) for i in self.pred[0].boxes.cls.tolist()]
        xyxy_list_int = [[round(num) for num in sublist] for sublist in self.pred[0].boxes.xyxy.tolist()]

        self.combined_image = draw_detections(self.img, xyxy_list_int, conf_list, cls_list_int, 0.4)

        self.result = cv2.cvtColor(self.combined_image, cv2.COLOR_BGR2BGRA)
        self.QtImg = QtGui.QImage(
            self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
        
        self.input.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.input.setScaledContents(True)  

        self.lineEdit.setText('图片检测成功！！！')

        cv2.imwrite('prediction.jpg', self.combined_image)
        
    # 视频检测
    def detect_video(self):
        self.timer.start()
        ret, frame = self.video.read()
        if not ret:
            self.timer.stop()
            self.video.release()
            self.out.release()
        else:
            name_list = []   

            self.comboBox.clear()

            # 检测每一帧
            self.pred = self.model.predict(source=frame, iou=self.numiou, conf=self.numcon)  # save plotted images
            preprocess_speed = self.pred[0].speed['preprocess']
            inference_speed = self.pred[0].speed['inference']
            postprocess_speed = self.pred[0].speed['postprocess']
            self.lineEdit_detect_time.setText(str(round((preprocess_speed + inference_speed + postprocess_speed) / 1000, 2)))
            self.lineEdit_detect_object_nums.setText(str(self.pred[0].boxes.conf.shape[0]))

            self.results = self.pred[0].boxes.xyxy.tolist()

            if self.pred[0].boxes.conf.shape[0]:
                for i in range(self.pred[0].boxes.conf.shape[0]):
                    self.comboBox.addItem('目标' + str(i + 1))

            # 画图
            conf_list = self.pred[0].boxes.conf.tolist()
            cls_list_int = [int(i) for i in self.pred[0].boxes.cls.tolist()]
            xyxy_list_int = [[round(num) for num in sublist] for sublist in self.pred[0].boxes.xyxy.tolist()]

            self.combined_image = draw_detections(frame, xyxy_list_int, conf_list, cls_list_int, 0.4)

            # 写视频
            self.out.write(self.combined_image)

            self.result_frame = cv2.cvtColor(self.combined_image, cv2.COLOR_BGR2BGRA)
            self.QtImg = QtGui.QImage(
                self.result_frame.data, self.result_frame.shape[1], self.result_frame.shape[0], QtGui.QImage.Format_RGB32)
            
            self.input.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            self.input.setScaledContents(True)  # 自适应界面大小
            self.lineEdit.setText('正在检测视频！！！')

    def suspend_video(self):
        self.timer.blockSignals(False)
        if self.timer.isActive() == True and self.num_stop % 2 == 1:
            self.button_video_suspend.setText(u'继续视频检测')  # 当前状态为暂停状态
            self.num_stop = self.num_stop + 1  # 调整标记信号为偶数
            self.timer.blockSignals(True)
        else:
            self.num_stop = self.num_stop + 1
            self.button_video_suspend.setText(u'暂停视频检测')
    # 暂停视频的检测
    def stop_video(self):
        if self.num_stop % 2 == 0:
            self.video.release()
            self.out.release()
            self.input.setPixmap(QPixmap())
            self.input.setScaledContents(True)

            self.button_video_suspend.setText(u'暂停视频检测')
            self.num_stop = self.num_stop + 1
            self.timer.blockSignals(False)
            self.lineEdit_detect_time.clear()
            self.lineEdit_detect_object_nums.clear()
            self.lineEdit_xmin.clear()
            self.lineEdit_ymin.clear()
            self.lineEdit_xmax.clear()
            self.lineEdit_ymax.clear()
            self.lineEdit.clear()
        else:
            self.video.release()
            self.out.release()
            self.input.setPixmap(QPixmap())
            self.input.setScaledContents(True)
            # self.output.clear()
            self.timer.blockSignals(False)
            self.lineEdit_detect_time.clear()
            self.lineEdit_detect_object_nums.clear()
            self.lineEdit_xmin.clear()
            self.lineEdit_ymin.clear()
            self.lineEdit_xmax.clear()
            self.lineEdit_ymax.clear()
            self.lineEdit.clear()
    # 关闭图片   
    def stop_image(self):
        self.input.setPixmap(QPixmap())
        self.input.setScaledContents(True)
        self.lineEdit_detect_time.clear()
        self.lineEdit_detect_object_nums.clear()
        self.lineEdit_xmin.clear()
        self.lineEdit_ymin.clear()
        self.lineEdit_xmax.clear()
        self.lineEdit_ymax.clear()
        self.comboBox.clear()
        self.lineEdit.clear()

    def export_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.OutputDir, _ = QFileDialog.getSaveFileName(
            self,  # 父窗口对象
            "导出图片",  # 标题
            r".",  # 起始目录
            "图片类型 (*.jpg *.jpeg *.png *.bmp)",  # 选择类型过滤项，过滤内容在括号中
            options=options
        )

        if self.OutputDir == "":
            QtWidgets.QMessageBox.warning(self, '提示', '请先选择图片保存的位置')
        else:
            try:
                dialog = QFileDialog(self, "Save image", self.OutputDir)
                dialog.resize(800, 600)
                dialog.close()
                cv2.imwrite(self.OutputDir, self.combined_image)
                QtWidgets.QMessageBox.warning(self, '提示', '导出成功!')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, '提示', '请先完成识别工作')
                print(e)

    def export_videos(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.OutputDirs, _ = QFileDialog.getSaveFileName(
            self,  # 父窗口对象
            "导出视频",  # 标题
            r".",  # 起始目录
            "图片类型 (*.mp3 *.mp4 *.gif *.avi)",  
            options=options
        )
        if self.OutputDirs == "":
            QtWidgets.QMessageBox.warning(self, '提示', '请先选择视频保存的位置')
        else:
            self.out.release()
            try:
                dialog = QFileDialog(self, "Save video", self.OutputDirs)
                dialog.resize(800, 600)
                dialog.close()
                shutil.copy(str(ROOT) + '/prediction.mp4', self.OutputDirs)
                QtWidgets.QMessageBox.warning(self, '提示', '导出成功!')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, '提示', '请先完成识别工作')
            
    def ValueChange(self):
        self.numcon = self.con_slider.value() / 100.0
        self.numiou = self.iou_slider.value() / 100.0
        self.con_number.setValue(self.numcon)
        self.iou_number.setValue(self.numiou)
    
    def Value_change(self):
        num_conf = self.con_number.value()
        num_ious = self.iou_number.value()
        self.con_slider.setValue(int(num_conf * 100))
        self.iou_slider.setValue(int(num_ious * 100))
        self.numcon = num_conf
        self.numiou = num_ious
        
    def value_change_comboBox(self):
        self.lineEdit_xmin.clear()
        self.lineEdit_ymin.clear()
        self.lineEdit_xmax.clear()
        self.lineEdit_ymax.clear()
        object = self.comboBox.currentText()
        if object:
            object_number_str = object[-1]
            object_number_int = int(object_number_str)
            object_number_index = object_number_int - 1
            if self.results:
                self.lineEdit_xmin.setText(str(int(self.results[object_number_index][0])))
                self.lineEdit_ymin.setText(str(int(self.results[object_number_index][1])))
                self.lineEdit_xmax.setText(str(int(self.results[object_number_index][2])))
                self.lineEdit_ymax.setText(str(int(self.results[object_number_index][3])))

    def open_camera(self):
        self.lineEdit.setText("打开摄像头中...")
        self.camera = cv2.VideoCapture(0)
        if self.camera.isOpened():
            self.lineEdit.setText("成功打开摄像头！")
            self.timer_c.start(30)
    # 检测摄像头
    def detect_camera(self):
        ret, frame = self.camera.read()
        if ret:
            result_input = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            self.QtImg_input = QtGui.QImage(
                result_input.data, result_input.shape[1], result_input.shape[0], QtGui.QImage.Format_RGB32)
            self.input.setPixmap(QtGui.QPixmap.fromImage(self.QtImg_input))
            self.input.setScaledContents(True)

            if self.running:
                name_list = []
                self.comboBox.clear()
                self.pred = self.model.predict(source=frame, conf=self.numcon, iou=self.numiou)  # save plotted images
                preprocess_speed = self.pred[0].speed['preprocess']
                inference_speed = self.pred[0].speed['inference']
                postprocess_speed = self.pred[0].speed['postprocess']
                self.lineEdit_detect_time.setText(str(round((preprocess_speed + inference_speed + postprocess_speed) / 1000, 2)))
                self.lineEdit_detect_object_nums.setText(str(self.pred[0].boxes.conf.shape[0]))

                self.results = self.pred[0].boxes.xyxy.tolist()

                if self.pred[0].boxes.conf.shape[0]:
                    for i in range(self.pred[0].boxes.conf.shape[0]):
                        self.comboBox.addItem('目标' + str(i + 1))

                
                conf_list = self.pred[0].boxes.conf.tolist()
                cls_list_int = [int(i) for i in self.pred[0].boxes.cls.tolist()]
                xyxy_list_int = [[round(num) for num in sublist] for sublist in self.pred[0].boxes.xyxy.tolist()]

                self.combined_image = draw_detections(frame, xyxy_list_int, conf_list, cls_list_int, 0.4)

                self.result_frame = cv2.cvtColor(self.combined_image, cv2.COLOR_BGR2BGRA)
                self.QtImg = QtGui.QImage(
                    self.result_frame.data, self.result_frame.shape[1], self.result_frame.shape[0], QtGui.QImage.Format_RGB32)
                
                self.input.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                self.input.setScaledContents(True)  # 自适应界面大小

                self.lineEdit.setText('正在使用摄像头进行检测！！！')
                
        else:
            self.timer_c.stop()
            self.camera.release()
            self.camera = None
    # 关闭摄像头
    def close_camera(self):
        self.running = False
        self.camera = None
        self.timer_c.stop()
        self.input.setPixmap(QPixmap())
        self.input.setScaledContents(True)

        self.lineEdit.setText("已关闭摄像头！")
        self.lineEdit_detect_time.clear()
        self.lineEdit_detect_object_nums.clear()
        self.lineEdit_xmin.clear()
        self.lineEdit_ymin.clear()
        self.lineEdit_xmax.clear()
        self.lineEdit_ymax.clear()

    def detect_camera_running(self):
        self.running = True

    def bind_slots(self):
        self.buttton_image_select.clicked.connect(self.open_image)
        self.buttton_video_select.clicked.connect(self.open_video)
        self.button_weight_select.clicked.connect(self.load_model)
        self.button_weight_init.clicked.connect(self.init_model)
        self.button_image_detect.clicked.connect(self.detect_begin)
        self.button_image_show.clicked.connect(self.detect_show)
        self.button_video_detect.clicked.connect(self.detect_video)
        self.button_video_suspend.clicked.connect(self.suspend_video)
        self.button_video_stop.clicked.connect(self.stop_video)
        self.button_image_stop.clicked.connect(self.stop_image)
        self.button_image_export.clicked.connect(self.export_images)
        self.button_video_export.clicked.connect(self.export_videos)
        self.con_slider.valueChanged.connect(self.ValueChange)
        self.iou_slider.valueChanged.connect(self.ValueChange)
        self.con_number.valueChanged.connect(self.Value_change)
        self.iou_number.valueChanged.connect(self.Value_change)
        self.comboBox.currentTextChanged.connect(self.value_change_comboBox)
        self.timer.timeout.connect(self.detect_video)
        self.button_camera_start.clicked.connect(self.open_camera)
        self.button_camera_stop.clicked.connect(self.close_camera)
        self.button_camera_detect.clicked.connect(self.detect_camera_running)

    def init_icons(self):
        self.label_weight_select.setPixmap(QPixmap('icons/weight.png'))
        self.label_weight_select.setScaledContents(True)
        self.label_weight_init.setPixmap(QPixmap('icons/init.png'))
        self.label_weight_init.setScaledContents(True)

        self.input.setPixmap(QPixmap())
        self.input.setScaledContents(True)

        self.label_main.setPixmap(QPixmap("bg.png"))
        self.label_main.setScaledContents(True)


        self.button_weight_select.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_weight_init.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_image_detect.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_image_export.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.buttton_image_select.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_image_show.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_image_stop.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_video_detect.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_video_export.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_video_stop.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_video_suspend.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.buttton_video_select.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_camera_detect.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_camera_start.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        self.button_camera_stop.setStyleSheet("QPushButton:pressed { background-color: rgb(135,206,250); }")
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle('基于YOLOV8和Pyside6的安全帽检测系统')
    window.setFixedSize(window.size())
    window.show()
    app.exec()