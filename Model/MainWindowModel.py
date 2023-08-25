import sys
from pathlib import Path
import threading
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QMainWindow, QFileSystemModel
from PyQt5.uic import loadUi

from Controller.MainWindowController import ProcessingProductImage, TrainingInputData_click, PredictClick, \
    LoadTrainedModel_click, Camera_Processing
from Helper.Errors import show_error
from Helper.FileHelper import sliderChangeValue, sliderResizeImage
from Helper.MyPathFunctions import GetCWD
from Helper.SystemHelper import ExitProgram
from Model import GlobalVariables
from Model.GlobalVariables import mainWindowUI, defaultPath


class MainWindowClass(QMainWindow):
    def __init__(self):
        try:
            super(MainWindowClass, self).__init__()
            loadUi(mainWindowUI, self)
            self.btnClose.clicked.connect(lambda: ExitProgram())
            self.selectedfile = ''

            # ---------------TreeFolder management
            savedPath = Path(defaultPath)
            if savedPath.exists():
                path = defaultPath
            else:
                path = str(GetCWD())

            self.dirModel = QFileSystemModel()
            self.dirModel.setRootPath(path)
            self.dirModel.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)

            self.fileModel = QFileSystemModel()
            self.fileModel.setFilter(QDir.NoDotAndDotDot | QDir.AllEntries)

            self.treeview.setModel(self.dirModel)
            self.listview.setModel(self.fileModel)

            self.treeview.setRootIndex(self.dirModel.index(path))
            self.listview.setRootIndex(self.fileModel.index(path))

            self.treeview.clicked.connect(self.tree_folder_clicked)
            self.listview.clicked.connect(self.list_file_clicked)

            self.btnUp.clicked.connect(self.GoUpper)

            # Slider change value
            self.slider1.valueChanged.connect(
                lambda: sliderChangeValue(self.slider1, self.slider2, self.lblPreview, self.lblFeature,
                                          self.slider_resize))

            self.slider2.valueChanged.connect(
                lambda: sliderChangeValue(self.slider1, self.slider2, self.lblPreview, self.lblFeature,
                                          self.slider_resize))

            self.slider_resize.valueChanged.connect(
                lambda: sliderResizeImage(self.slider_resize, self.lbl_resize_label, self.lblPreview, self.lblFeature))


            self.btnConnectCamera.clicked.connect(self.ConnectCamera)
            #Button predict

            #Button load model click:

        except Exception as e:
            print(e)


    def ConnectCamera(self):
        try:

            x = threading.Thread(target=Camera_Processing, args=(self,))
            x.start()
            # Camera_Processing(self)
        except Exception as e:
            show_error(e)
        finally:
            pass



    def list_file_clicked(self, index):
        try:
            indexItem = self.fileModel.index(index.row(), 0, index.parent())
            fileName = self.fileModel.fileName(indexItem)
            path = self.dirModel.filePath(indexItem)
            self.selectedfile = path
            ProcessingProductImage(path, self)
        except Exception as e:
            print(e)

    def tree_folder_clicked(self, index):
        try:
            indexItem = self.dirModel.index(index.row(), 0, index.parent())
            path = self.dirModel.filePath(indexItem)
            self.listview.setRootIndex(self.fileModel.setRootPath(path))
            self.lblCurrentPath.setText(path)
        except Exception as e:
            print(e)


    def GoUpper(self):
        try:
            index = self.treeview.currentIndex()
            parent = index.parent().parent()
            parentPath = self.dirModel.filePath(parent)
            self.treeview.setRootIndex(self.dirModel.setRootPath(parentPath))
        except Exception as e:
            print(e)
