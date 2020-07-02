import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton,QLineEdit
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import os

def window():
   app = QApplication(sys.argv)
   widget = QWidget()
   
   button1 = QPushButton(widget)
   button1.setText("Button1")
   button1.setGeometry(100,100,150,50)
   button1.move(100,0)
   button1.clicked.connect(button1_clicked)
   label1 =QLineEdit()
   
   button2 = QPushButton(widget)
   button2.setText("Button2")
   button2.setGeometry(100,100,150,50)
   button2.move(100,100)
   button2.clicked.connect(button2_clicked)
   label2 =QLineEdit()
   
   button3 = QPushButton(widget)
   button3.setText("Button3")
   button3.setGeometry(100,100,150,50)
   button3.move(100,200)
   button3.clicked.connect(button3_clicked)
   label3 =QLineEdit()
   
   widget.setGeometry(100,100,500,500)
   widget.setWindowTitle("PyQt5 Button Click Example")
   widget.show()
   sys.exit(app.exec_())


def button1_clicked():
   a="Button 1 clicked start"
   print(a)
   cmd="python 29subat.py"
   os.system(cmd)

def button2_clicked():
   b="Button 2 clicked stop"
   print(b)
   os.system("q")
  

def button3_clicked():
   c="Button 3 clicked call "
   print(c)
   
   
if __name__ == '__main__':
   window()