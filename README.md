柑橘成熟度分割数据集，基于YOLOv11的柑橘成熟度实例分割系统 柑橘成熟度分割数据集，![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/38022de3e45746bf9e8500f8e6f7806f.png)
7417张，yolo，voc，coco三种标注方式
数据集统计信息![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8acf9980f89142d4834fdac8b9773494.png)

====================
图像尺寸:640*640（有添加噪声处理）
类别数量: 3 类
训练集图像:6492; 验证集图像:614; 测试集图像:311
总图像数: 7417
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c23b5899d3d542beb32047ba2c017909.png)

类别分布:
类别名称 | 包含该类别的图像数 | 总标注数
----------------------------------------
Full Ripe 全熟：2616 | 3322
Red Scale 部分成熟：2593 | 2819
Unripe 未熟：2571 | 3244


模型代码：模型训练使用yolov11n-seg训练，80个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a000334d99c34569abafc19e98a1c686.png)

qt界面：运行界面采用pyqt编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像，下图![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a83699c5b0164080a125006a33a4715d.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/58738090609c49adbf544c2a75094e6d.png)
水印已清除，下面附上 **YOLOv11‑seg 柑橘实例分割完整可运行源码**
# 1、数据集配置文件 citrus.yaml
```yaml
path: ./datasets/citrus
train: images/train
val: images/val
test: images/test

names:
  0: Full_Ripe
  1: Red_Scale
  2: Unripe
```
# 2、分割训练代码 train_seg.py![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/35f3ef2deb22470d9ee7e8d8038e3c3a.png)

```python
from ultralytics import YOLO
#加载分割权重
model = YOLO("yolo11n-seg.pt")

results = model.train(
    data="citrus.yaml",
    epochs=80,
    imgsz=640,
    batch=16,
    device=0,
    workers=0
)
```
# 3、图像分割推理 predict_seg.py
```python
from ultralytics import YOLO

model = YOLO("./runs/segment/train/weights/best.pt")

res = model.predict(
    source="test.jpg",
    conf=0.5,
    save=True
)
```
# 4、PyQt5可视化分割GUI界面代码 main.py![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8a976c381c5648b3a356fc3c0f81b14a.png)

```python
import sys
import cv2
from PyQt5.QtWidgets import (QApplication,QMainWindow,QPushButton,QFileDialog,QLabel,QTableWidget,QTableWidgetItem,QHeaderView)
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class CitrusSegWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的柑橘成熟度实例分割系统")
        self.resize(1600,900)
        self.model = YOLO("./runs/segment/train/weights/best.pt")
        self.init_ui()

    def init_ui(self):
        #原图显示
        self.label_origin = QLabel("原始图像",self)
        self.label_origin.setGeometry(20,60,750,700)
        #分割结果图
        self.label_result = QLabel("分割结果",self)
        self.label_result.setGeometry(800,60,750,700)
        #按钮
        self.btn_img = QPushButton("选择图片",self)
        self.btn_img.setGeometry(20,10,140,40)
        self.btn_img.clicked.connect(self.detect_image)
        self.btn_video = QPushButton("选择视频",self)
        self.btn_video.setGeometry(180,10,140,40)
        self.btn_cam = QPushButton("摄像头",self)
        self.btn_cam.setGeometry(340,10,140,40)
        #结果表格
        self.table = QTableWidget(self)
        self.table.setGeometry(20,780,1530,100)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["序号","类别","置信度","坐标"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def detect_image(self):
        path,_ = QFileDialog.getOpenFileName(self,"打开图片","","*.jpg;*.png")
        if not path:
            return
        res = self.model.predict(source=path,conf=0.5,save=False)[0]
        img_origin = cv2.imread(path)
        img_result = res.plot()
        self.show_img(img_origin,self.label_origin)
        self.show_img(img_result,self.label_result)
        #填充表格
        self.table.setRowCount(0)
        for idx,box in enumerate(res.boxes):
            row = self.table.rowCount()
            self.table.insertRow(row)
            cls_name = self.model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            xyxy = [round(float(x),1) for x in box.xyxy[0]]
            self.table.setItem(row,0,QTableWidgetItem(str(idx+1)))
            self.table.setItem(row,1,QTableWidgetItem(cls_name))
            self.table.setItem(row,2,QTableWidgetItem(f"{conf:.2f}"))
            self.table.setItem(row,3,QTableWidgetItem(str(xyxy)))

    def show_img(self,img,label):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,c = img.shape
        qimg = QImage(img.data,w,h,c*w,QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(),Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CitrusSegWindow()
    win.show()
    sys.exit(app.exec_())
```
# 环境安装命令
```bash
pip install ultralytics pyqt5 opencv-python
```

