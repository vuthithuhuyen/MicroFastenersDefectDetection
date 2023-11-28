import os
import time
from pathlib import Path

# from joblib import dump, load
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QLabel, QMainWindow

from Helper.Errors import show_error
from Helper.OpenCVHelper import IsImage, RawImageToArray, ReadImage, rescale_frame, Detect_Edge
from Helper.PyQTHelper import QLabelDisplayImage, showDialog, showErrDialog
from Model import GlobalVariables
from Model.GlobalVariables import thres1, thres2, train_class, svt_model
from Model.RunningTime import MyRunningTime
from yolov8 import SVT_classification


# from Helper.SVMHelper import SVMTrainer


# from Model.RunningTime import MyRunningTime


# Đọc webcam rồi xử lý:
def Camera_Processing(self: QMainWindow):
    stream = None
    try:
        # Step 1: Kết nối đến camera bằng Vidgear

        camera_source = self.cmbCameraID.currentIndex()
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            print(f'-------CamerID {camera_source} is closed')
            return

        count = 0
        while True:
            ret, frame = cap.read()
            count += 1
            if frame is None or self.chkStop.isChecked():
                cap.release()
                cv2.destroyAllWindows()
                break

            # Step 2: Chuyển thông tin sang GUI (Giống như tree view on click)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ProcessingProductImage(self=self, frame=frame)
            # ProcessingProductImage(frame=frame, self=self)

    except Exception as e:

        show_error(e)
    finally:
        pass


# Bấm vào danh sách file trên list. Nếu là file ảnh thì hiển thị
def ProcessingProductImage(filename='', self: QMainWindow = None, frame=None):
    if frame is None:
        if not IsImage(filename):
            print(f'{filename} is not an image file')
            return
    try:

        slider1, slider2, lblPreview, lblFeature = self.slider1, self.slider2, self.lblPreview, self.lblFeature
        lblPredictResult, slider_resize = self.lblPredictResult, self.slider_resize

        img = None
        # Đoc từ file ảnh
        if frame is None:
            GlobalVariables.currentImage = filename
            imageFile = GlobalVariables.currentImage
            if not os.path.isfile(imageFile):
                print(f'----not an image file: {imageFile}')
                return
            img = ReadImage(imageFile)
        else:  # Đọc từ Webcam
            img = frame

        img_resized = rescale_frame(img, slider_resize.value())

        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        GlobalVariables.thres1 = slider1.value()
        GlobalVariables.thres2 = slider2.value()
        img_edges = Detect_Edge(gray, GlobalVariables.thres1, GlobalVariables.thres2)
        if frame is not None:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

        QLabelDisplayImage(lblPreview, img_resized)
        QLabelDisplayImage(lblFeature, img_edges)


        # Prediction and Classification
        predict_img = SVT_classification(svt_model, filename)
        predict_img = rescale_frame(predict_img, slider_resize.value())

        # if frame is None: # Nếu đọc từ file ảnh thì cần convert lại sang BGR
        #     predict_img = cv2.cvtColor(predict_img, cv2.COLOR_RGB2BGR)

        QLabelDisplayImage(lblPredictResult, predict_img)



        # self.lbl_product_result.setText(result)
        # update_output_label(self, product_ok)

    except Exception as e:
        print(e)


# Cập nhật nhãn đếm OK và NG
def update_output_label(self: QMainWindow, status: bool):
    try:

        current_label = self.lbl_ok.text() if status else self.lbl_not_good.text()
        try:
            current_label = int(current_label)
        except Exception as e:
            current_label = 0
        current_label += 1

        # self.lbl_ok.setText(str(current_label)) if status else self.lbl_not_good.setText(str(current_label))

    except Exception as e:
        show_error(e)
    finally:
        pass



# Đọc dữ liệu file từ danh sách trong data frame -> Chuyển vào train_data, train_label
def ListFileInFrame2TrainData(df: pd):
    list_data, listLabel = [], []
    executeTime = MyRunningTime()

    for index, row in df.iterrows():
        print(f'Reading image... {row[1]}')
        label = row[0]
        img_array = RawImageToArray(row[2], thres1, thres2)

        listLabel.append(label)
        list_data.append(img_array)

    features = len(list_data[0])
    train_data = np.empty((0, features), int)
    train_label = np.array(listLabel)

    count = 0
    for row in list_data:
        try:
            count += 1
            row = row.reshape(-1, features)
            train_data = np.append(train_data, row, axis=0)
        except Exception as e:
            print(f'Error at index {count} shape: {row.shape}')
            print(e)

    executeTime.CalculateExecutedTime()
    print(f'finish reading file in {executeTime.runningTime}')
    return train_data, train_label



