from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QLabel, QLayout, QVBoxLayout, QBoxLayout, QHBoxLayout,  QWidget, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPainter, QPixmap, QColor, QPen

import numpy as np
import matplotlib.pyplot as plt

from screeninfo import get_monitors

from nn import *

class App(QWidget):
    def __init__(self):
        super().__init__()

        self.scale = self.get_screen_res()

        self.x = 100 * self.scale
        self.y = 100 * self.scale
        self.width = 1024 * self.scale
        self.height = 600 * self.scale

        self.init_setup()


    def init_setup(self):
        self.setGeometry(self.x, self.y, self.width, self.height)
        
        # Contains 4 Pixmaps 
        self.textbox = []

        # Creates 4 Pixmaps and sends it to self.textbox
        for i in range(4):
            
            canvas = Pixmap(int(self.width/4) - 20, int(self.width/4) - 20)
            canvas.fill((QColor('black')))
            
            canvas_label = Label(canvas, self.scale)
            self.textbox.append(canvas_label)
        
        self.hbox1 = QHBoxLayout()
        for i in range(4):
            self.hbox1.addWidget(self.textbox[i])


        self.submit_button = QPushButton("Submit", self)
        self.submit_button.clicked.connect(self.submit)

        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear)
        self.hbox2 = QHBoxLayout()
        self.hbox2.addWidget(self.clear_button.setGeometry(self.width/2 - (150 + 60)*self.scale, self.height-100*self.scale, 150*self.scale, 50*self.scale))
        self.hbox2.addWidget(self.submit_button.setGeometry(self.width/2 - (0 - 60)*self.scale, self.height-100*self.scale, 150*self.scale, 50*self.scale))

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hbox1)

        self.vbox.addLayout(self.hbox2)
        self.vbox.setAlignment(Qt.AlignTop | Qt.AlignCenter)

        self.setLayout(self.vbox)

        self.show()



    def get_screen_res(self):
        return get_monitors()[0].width/1920

    # def mousePressEvent(self, event):
    #     if event.button() == Qt.LeftButton:
    #         print("Mouse Pressed")



    def resizeEvent(self, event):
        height = event.size().height()
        width = event.size().width()

        self.clear_button.move(width/2 - (150 + 60)*self.scale, height-100*self.scale)
        self.submit_button.move(width/2 - (0 - 60)*self.scale, height-100*self.scale)
        QWidget.resizeEvent(self, event)


    def clear(self):
        for i in range(4):
            self.textbox[i].clear()

    def submit(self):
        print("act")
        numbers = np.zeros((4, int(self.width/4-20), int(self.width/4-20), 4))
        all_interacted = True
        for i in range(4):
            if not self.textbox[i].interacted:
                all_interacted = False
            pxmp = self.textbox[i].getPixmap()
            width = pxmp.size().width()
            height = pxmp.size().height()
            print(pxmp.size())

            qimg = pxmp.toImage()
            byte_str = qimg.bits()
            byte_str.setsize(height*width*4)

            img = np.frombuffer(byte_str, dtype=np.uint8).reshape((width, height, 4))
            numbers[i, :, :, :] = img
            self.textbox[i].interacted = False
        if all_interacted:
            #print(numbers)
            predict_nums(numbers)
        else:
            print("Not Interacted")

        
        


        

class Pixmap(QPixmap):
    def __init__(self, width, height):
        super().__init__(width, height)


class Label(QLabel):
    def __init__(self, image, scale):
        super().__init__()
        self.draw = False
        self.interacted = False
        self.prev = QPoint()
        self.image = image
        self.setPixmap(self.image)
        self.scale = scale

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.image is not None:
            painter.drawPixmap(self.image.rect(), self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            print("Mouse Pressed")
            self.draw = True
            self.interacted = True
            self.prev = event.pos()
    def mouseMoveEvent(self, event):
        if event.buttons() and self.draw and Qt.LeftButton:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.white, 25*self.scale, Qt.SolidLine))
            painter.drawLine(self.prev, event.pos())
            self.prev = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.draw = False

    def getPixmap(self):
        return self.image

    def clear(self):
        self.image.fill((QColor('black')))
        self.setPixmap(self.image)    
        self.interacted = False







if __name__ == "__main__":
    app = QApplication([])
    window = App()

    app.exec_()

