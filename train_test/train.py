## Input Parameter Start Here ##
path_train_img = 'E:/19-9-2022/dev1/cropped/train/*.jpg'  ## Path train images
print (path_train_img)
## Input Parameter end Here ##

from contextlib import contextmanager
from io import StringIO
from statistics import mode
from matplotlib import image
# from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
import sys
from time import sleep

from PIL import Image
import io
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import cv2

sys.path.append ('C:/Users/41162395/anomaly-python/ind_knn_ad/indad') #('./indad') # 
from data import MVTecDataset, StreamingDataset
from models import SPADE, KNNExtractor
from data import IMAGENET_MEAN, IMAGENET_STD

import glob

image_train = []

def load_image():
    for item in glob.glob(path_train_img):
        img = Image.open(item)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        img_byte = buf.getvalue()
        image_train.append(img_byte)


def main():
    if len(image_train) > 2:
        # test dataset will contain 1 test image
        train_dataset = StreamingDataset()
        # train images
        for training_image in image_train:
            # training_image_re = training_image.resize((224,224))
            # buf = io.BytesIO(training_image)
            # training_image.save(buf, format='JPEG')
            # bytes_data = buf.getvalue()
            # bytes_data = training_image.getvalue()
            train_dataset.add_pil_image(
                Image.open(io.BytesIO(training_image))
            )
    else:
        print('Error')
    model = SPADE(
                    k=3,
                    backbone_name="efficientnet_b0",
                )
    # if image_test is not None:
    #     # model = torch.load('weight_normal.pt')
    #     # model.load_state_dict(torch.load('weight.pt'))
    # model.eval()

    # for par in model.parameters():
    #     print(par)
    model.fit(DataLoader(train_dataset))
    # torch.save(model.state_dict(), 'weight.pt')
    torch.save(model, 'weight_normal.pt')


if __name__ == "__main__":
    load_image()
    main()