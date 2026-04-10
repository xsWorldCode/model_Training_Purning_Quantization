import os
import sys
import cv2
import torch
import time
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QFrame, QLabel, QPushButton, QStatusBar, QWidget
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QThread, Signal, QRect

# --- 1. UI 布局结构 ---
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(850, 420)
        self.centralwidget = QWidget(MainWindow)
        
        # 左邊的原圖
        self.input = QLabel(self.centralwidget)
        self.input.setObjectName(u"input")
        self.input.setGeometry(QRect(30, 40, 360, 270))
        self.input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input.setStyleSheet("border: 1px solid #555; background-color: #1e1e1e; color: #888;")
        
        # 右邊的結果圖
        self.output = QLabel(self.centralwidget)
        self.output.setObjectName(u"output")
        self.output.setGeometry(QRect(460, 40, 360, 270))
        self.output.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output.setStyleSheet("border: 2px solid #00ff00; background-color: #1e1e1e; color: #00ff00;")

        self.detect_image = QPushButton(self.centralwidget)
        self.detect_image.setGeometry(QRect(30, 330, 360, 40))
        self.det_video = QPushButton(self.centralwidget)
        self.det_video.setGeometry(QRect(460, 330, 360, 40))

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle("Vision Studio Pro - RTX 5060 Ti Edition")
        self.input.setText("Original Image")
        self.output.setText("Detection Results")
        self.detect_image.setText("Select Image for Detection")
        self.det_video.setText("Start Real-time Video Detection")

# --- 2. 异步视频推理线程 ---
class VideoThread(QThread):
    data_signal = Signal(np.ndarray, np.ndarray, object, int)

    def __init__(self, model, source_path):
        super().__init__()
        self.model = model
        self.source_path = source_path
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(self.source_path)
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                # 关键：备份原图内存，防止 render 直接修改
                raw_frame = frame.copy()
                start_t = time.time()
                
                results = self.model(frame)
                res_img = results.render()[0] 
                
                dt = int((time.time() - start_t) * 1000)
                self.data_signal.emit(raw_frame, res_img, results, dt)
            else:
                break
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# --- 3. 主程序业务逻辑 ---
class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # 硬件与路径配置
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.repo_dir = r'F:\python_code\yolov5-master\yolov5-master'
        self.weights = os.path.join(self.repo_dir, 'yolov5s.pt')
        
        self.statusbar.showMessage("正在初始化 5060 Ti 加速引擎...")
        self.model = torch.hub.load(self.repo_dir, 'custom', path=self.weights, source='local', device=self.device)
        self.statusbar.showMessage(f"已就绪 | 设备: {self.device}")

        # 绑定点击查询功能
        self.output.installEventFilter(self)
        self.last_results = None
        self.video_thread = None

        self.detect_image.clicked.connect(self.process_image)
        self.det_video.clicked.connect(self.process_video)

    # --- 交互：点击右侧结果图查询详细信息 ---
    def eventFilter(self, source, event):
        if source == self.output and event.type() == event.Type.MouseButtonPress:
            pos = event.pos()
            self.handle_click_query(pos.x(), pos.y())
        return super().eventFilter(source, event)

    def handle_click_query(self, x_ui, y_ui):
        if self.last_results is None: return
        
        pixmap = self.output.pixmap()
        if not pixmap: return
        
        # 修复属性名兼容性 (ims vs imgs)
        try:
            img_h, img_w = self.last_results.ims[0].shape[:2]
        except AttributeError:
            img_h, img_w = self.last_results.imgs[0].shape[:2]
        
        # 坐标映射计算
        scale_x = img_w / pixmap.width()
        scale_y = img_h / pixmap.height()
        
        offset_x = (self.output.width() - pixmap.width()) / 2
        offset_y = (self.output.height() - pixmap.height()) / 2
        
        real_x = (x_ui - offset_x) * scale_x
        real_y = (y_ui - offset_y) * scale_y

        # 遍历检测框
        df = self.last_results.pandas().xyxy[0]
        for _, row in df.iterrows():
            if row['xmin'] <= real_x <= row['xmax'] and row['ymin'] <= real_y <= row['ymax']:
                info = f"目标: {row['name']} | 置信度: {row['confidence']:.2f} | 原始坐标: ({int(real_x)}, {int(real_y)})"
                self.statusbar.showMessage(info)
                return

    # --- UI 渲染工具 ---
    def display_frame(self, cv_img, label_obj):
        h, w, ch = cv_img.shape
        # BGR -> RGB 转换
        q_img = QImage(cv_img.data, w, h, ch * w, QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        label_obj.setPixmap(pixmap.scaled(label_obj.width(), label_obj.height(), Qt.AspectRatioMode.KeepAspectRatio))

    # --- 推理入口 ---
    def process_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择待测图片", "", "Images (*.jpg *.png *.bmp)")
        if file_path:
            raw_img = cv2.imread(file_path)
            # 必须备份原图，否则 render 会直接修改 raw_img
            clean_display = raw_img.copy() 
            
            start_t = time.time()
            self.last_results = self.model(raw_img)
            res_img = self.last_results.render()[0]
            dt = int((time.time() - start_t) * 1000)
            
            self.display_frame(clean_display, self.input) # 左边原图
            self.display_frame(res_img, self.output)      # 右边结果
            self.statusbar.showMessage(f"单图检测完成 | 耗时: {dt}ms | 可点击右侧画面查询物体详情")

    def process_video(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.det_video.setText("开启视频实时检测")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi)")
        if file_path:
            self.det_video.setText("停止检测")
            self.video_thread = VideoThread(self.model, file_path)
            self.video_thread.data_signal.connect(self.update_video_ui)
            self.video_thread.start()

    def update_video_ui(self, raw, res, results, dt):
        self.last_results = results
        self.display_frame(raw, self.input)
        self.display_frame(res, self.output)
        self.statusbar.showMessage(f"5060 Ti 动力全开 | 延迟: {dt}ms")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec())