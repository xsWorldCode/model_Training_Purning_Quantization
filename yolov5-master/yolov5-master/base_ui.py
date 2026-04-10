import sys
import os
import cv2
import torch  # 导入 torch 用于加载 YOLOv5
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer
from mainWindow_ui import Ui_MainWindow 

# 图像转换工具：将 OpenCV 的 BGR 格式转为 Qt 的 RGB 格式
def convertwQImage(img):
    # YOLOv5 render() 返回的是 BGR 格式，转换为 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape
    bytesPerLine = channel * width
    return QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 1. 初始化 UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # 2. 加载 YOLOv5 模型
        # 第一次运行会联网下载 yolov5s.pt，之后会从本地加载
        print("正在加载 YOLOv5 模型，请稍候...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
        print(">>> 模型加载成功！")

        # 3. 业务变量
        self.timer = QTimer() 
        self.cap = None       
        
        # 4. 绑定信号与槽
        self.bind_slots()
        print(">>> 界面已就绪，可以进行图片或视频检测")

    def bind_slots(self):
        # 绑定图片检测按钮
        self.ui.detect_image.clicked.connect(self.open_image)
        # 绑定视频检测按钮 (对应你 UI 中的 det_video)
        self.ui.det_video.clicked.connect(self.open_video)
        # 绑定定时器
        self.timer.timeout.connect(self.update_frame)

    # ========================== 核心预测逻辑 ==========================
    
    def image_pred(self, img_path):
        """对图片路径进行预测"""
        # 1. 推理
        results = self.model(img_path)
        # 2. 渲染（在图上画框并返回列表，取第 0 个结果）
        annotated_img = results.render()[0] 
        # 3. 转换为 QImage
        return convertwQImage(annotated_img)

    def video_pred(self, frame):
        """对视频的单帧 (numpy 数组) 进行预测"""
        # YOLOv5 直接支持传入 OpenCV 读取的帧
        results = self.model(frame)
        annotated_frame = results.render()[0]
        return convertwQImage(annotated_frame)

    # ========================== 业务操作逻辑 ==========================

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "./", "Images (*.jpg *.png *.jpeg)")
        if file_path:
            self.stop_video()
            print(f">>> 正在检测图片: {file_path}")
            
            # A. 左侧显示原始图片
            self.ui.input.setScaledContents(True)
            self.ui.input.setPixmap(QPixmap(file_path))
            
            # B. 右侧显示检测后的结果图片
            q_img = self.image_pred(file_path)
            self.ui.output.setScaledContents(True)
            self.ui.output.setPixmap(QPixmap.fromImage(q_img))
            print(">>> 图片检测完成")

    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "./", "Videos (*.mp4 *.avi *.mkv)")
        if file_path:
            self.stop_video()
            self.cap = cv2.VideoCapture(file_path)
            if self.cap.isOpened():
                print(f">>> 开始检测视频: {file_path}")
                # 启动定时器，每 30ms 刷新一帧
                self.timer.start(30) 
            else:
                print(">>> 错误：无法打开视频文件")

    def update_frame(self):
        """定时器回调：逐帧读取、检测并显示"""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 1. 左侧显示原视频帧
                raw_q = convertwQImage(frame)
                self.ui.input.setScaledContents(True)
                self.ui.input.setPixmap(QPixmap.fromImage(raw_q))
                
                # 2. 右侧显示 YOLOv5 检测后的视频帧
                pred_q = self.video_pred(frame)
                self.ui.output.setScaledContents(True)
                self.ui.output.setPixmap(QPixmap.fromImage(pred_q))
            else:
                self.stop_video()
                print(">>> 视频检测结束")

    def stop_video(self):
        """停止视频流并释放资源"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())