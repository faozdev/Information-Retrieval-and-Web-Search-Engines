# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AboutPage.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QDesktopServices

class About_Window(QMainWindow):
    def __init__(self):
        from PyQt5.QtGui import QFont

        super().__init__()

        self.label_2 = QLabel(self)
        self.label_2.setText("Recommendation System")
        self.label_2.setGeometry(50, 20, 231, 71)
        font_2 = QFont("MS Shell Dlg 2", 12)
        self.label_2.setFont(font_2)


        self.label_2_5 = QLabel(self)
        self.label_2_5.setText("Created by:")
        self.label_2_5.setGeometry(30, 180, 111, 61)
        font_2_5 = QFont("MS Shell Dlg 2", 10)
        self.label_2_5.setFont(font_2_5)

        self.label_3 = QLabel(self)
        self.label_3.setText("Fatih Özcan")
        self.label_3.setGeometry(140, 190, 111, 41)
        font_3 = QFont("MS Shell Dlg 2", 10)
        self.label_3.setFont(font_3)

        self.label_4 = QLabel(self)
        self.label_4.setText("Version 1.0")
        self.label_4.setGeometry(30, 130, 131, 21)
        font_4 = QFont("MS Shell Dlg 2", 10)
        self.label_4.setFont(font_4)

        self.label_5 = QLabel(self)
        self.label_5.setText("<a href='https://github.com/faozdev'>GitHub</a><br><br>")
        self.label_5.setGeometry(30, 260, 71, 41)
        font_5 = QFont("MS Shell Dlg 2", 10)
        self.label_5.setFont(font_5)

        # Bağlantıyı tıklanabilir yap
        self.label_5.setOpenExternalLinks(True)
        self.label_5.linkActivated.connect(self.linkiAc)
        self.resize(343, 435)
        self.setWindowTitle("About")

    def linkiAc(self, link):
        QDesktopServices.openUrl(QUrl(link))
   
    

    
#import icon_rc


if __name__ == "__main__":
    app = QApplication([])
    pencere = About_Window()
    pencere.show()
    app.exec_()
"""
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(343, 435)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(20, 20, 51, 51))
        self.graphicsView.setStyleSheet("border-image: url(:/newPrefix/info-circle-outline.png);")
        self.graphicsView.setObjectName("graphicsView")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(90, 10, 231, 71))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 180, 111, 61))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(140, 190, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(30, 130, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(40, 260, 71, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "About"))
        self.label.setText(_translate("MainWindow", "Recommendation System "))
        self.label_2.setText(_translate("MainWindow", "Created by:"))
        self.label_3.setText(_translate("MainWindow", "Fatih Özcan"))
        self.label_4.setText(_translate("MainWindow", "Version 1.0"))
        self.label_5.setText(_translate("MainWindow", "Github"))
#import icon_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
"""