# 客户端
# 导入必须的包
import socket
import cv2
import threading
import struct
import numpy


# 定义一个类，用于连接摄像头，并设置参数
class Camera_Connect_Object:
    # 初始化参数
    def __init__(self, D_addr_port=["", 9999]):
        self.resolution = [384, 288]
        self.addr_port = D_addr_port
        self.src = 888 + 15  # 双方确定传输帧数，（888）为校验值
        self.interval = 0  # 图片播放时间间隔
        self.img_fps = 15  # 每秒传输多少帧数

    # 设置socket连接
    def Set_socket(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # 连接摄像头
    def Socket_Connect(self):
        # 设置socket
        self.Set_socket()
        # 连接服务器
        self.client.connect(self.addr_port)
        # 打印服务器IP和端口
        print("IP is %s:%d" % (self.addr_port[0], self.addr_port[1]))

    # 接收图片
    def RT_Image(self):
        # 按照格式打包发送帧数和分辨率
        self.name = self.addr_port[0] + " Camera"
        self.client.send(struct.pack("lhh", self.src, self.resolution[0], self.resolution[1]))
        while (1):
            info = struct.unpack("lhh", self.client.recv(8))
            buf_size = info[0]  # 获取读的图片总长度
            if buf_size:
                try:
                    self.buf = b""  # 代表bytes类型
                    temp_buf = self.buf
                    while (buf_size):  # 读取每一张图片的长度
                        temp_buf = self.client.recv(buf_size)
                        buf_size -= len(temp_buf)
                        self.buf += temp_buf  # 获取图片
                        data = numpy.fromstring(self.buf, dtype='uint8')  # 按uint8转换为图像矩阵
                        self.image = cv2.imdecode(data, 1)  # 图像解码
                        cv2.imshow(self.name, self.image)  # 显示图片
                except:
                    pass;
                finally:
                    if (cv2.waitKey(10) == 27):  # 每10ms刷新一次图片，按‘ESC’（27）退出
                        self.client.close()
                        cv2.destroyAllWindows()
                        break

    # 获取图片数据
    def Get_Data(self, interval):
        showThread = threading.Thread(target=self.RT_Image)
        showThread.start()


if __name__ == '__main__':
    # 创建一个Camera_Connect_Object对象
    camera = Camera_Connect_Object()
    # 获取用户输入的IP地址
    camera.addr_port[0] = input("Please input IP:")
    # 将IP地址转换为元组
    camera.addr_port = tuple(camera.addr_port)
    # 连接socket
    camera.Socket_Connect()
    # 获取数据
    camera.Get_Data(camera.interval)