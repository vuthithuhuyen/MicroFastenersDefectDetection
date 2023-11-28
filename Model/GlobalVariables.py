import os
import pathlib
import numpy as np
from ultralytics import YOLO


from Helper.MyPathFunctions import GetCWD

appPath = GetCWD()
mainWindowUI = appPath / 'View' / 'mygui.ui'
train_class = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

defaultPath = str(appPath)
currentImage = ""
blurImage = False
thres1, thres2 = 70, 33



standardImgSize = (28, 28)
display_rescale = 50

training_size = (50, 21)
colors_list = ((0, 255, 0),  # green
               (0, 0, 255),  # red
               (0, 100, 255),  # orange
               (0, 220, 255),  # yellow
               (0, 0, 255),  # blue
               (0, 0, 255),  # indigo
               (0, 255, 255)  # violet
               )

# If the intersection area is larger than this ratio, then => belongs to the corresponding defect type.
percent_defect = {'Circle1': 0.6, 'Circle2': 0.3, 'Circle3': 0.1}
svt_model = YOLO('SVTm-best.pt')