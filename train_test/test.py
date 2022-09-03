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
import torchvision.transforms as T
import cv2

transform = T.ToPILImage()
# sys.path.append('./indad')
sys.path.append('C:/Users41162395/anomaly/ind_knn_ad/indad')
from data import MVTecDataset, StreamingDataset
from models import SPADE, KNNExtractor
from data import IMAGENET_MEAN, IMAGENET_STD

import glob

## Input Parameter Start Here ##
acm = 120   # Anomaly Score Max
path_anomaly_image = r'E:/2-9-2022/black-not/cropped/reject/'         # Path for store anomaly images
path_test_img_dir = r'E:/2-9-2022/black-not/cropped/'  # Path for image under test
## Input Parameter End Here ##

path_test_img_dir_len = len(path_test_img_dir)
path_test_img = str(path_test_img_dir) + '*.png'

image_test = []
myfilename = []
def load_image():
    for item in glob.glob(path_test_img):
        img = Image.open(item)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        img_byte = buf.getvalue()
        image_test.append(img_byte)
        myfilename.append(item)
    # print(myfilename)
    print(len(myfilename))
    for i in range(len(myfilename)):
        # print (myfilename[i])
        x = str(myfilename[i])
        y = len(x)
        myfilename[i] = x[path_test_img_dir_len:y] #E:/29-8-2022/valid\1661832970846.jpg

def main():
    if  len(image_test) > 0:
            # test dataset will contain 1 test image
            test_dataset = StreamingDataset()
            # test image
            for test_image in image_test:
                # test_image_re = test_image.resize((224,224))
                # buf = io.BytesIO(test_image)
                # test_image.save(buf, format='JPEG')
                # bytes_data = buf.getvalue()
                # bytes_data = test_image.getvalue()
                test_dataset.add_pil_image(
                    Image.open(io.BytesIO(test_image))
                )
    else:
        print('Error')
    model = SPADE(
                    k=3,
                    backbone_name="efficientnet_b0",
                )
    if image_test is not None:
        model = torch.load('weight_normal.pt')
    model.eval()
    cnt = 0
    for img in test_dataset:
        sample, *_ = img
        img_lvl_norm,pixel_lvl_norm = model.predict(sample.unsqueeze(0))
        score = pixel_lvl_norm.min(),pixel_lvl_norm.max()
        # print("score min : {:.0f}".format(score[0]))
        print(str(cnt)+" Score max : {:.0f}   ".format(score[1])+str(myfilename[cnt]))
        if (score[1]) > acm:
            #mynp = sample.unsqueeze(0).cpu().detach().numpy()
            mynp = sample.unsqueeze(0).cpu().numpy()
            grayImage = mynp.reshape(3,224, 224)
            t = grayImage.transpose() * 80 + 120
            #print(t.shape)
            gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            image = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
            # cv2.imwrite(r"E:/29-8-2022/anomaly_img/"+str(cnt)+".jpg",image)
            cv2.imwrite(path_anomaly_image+'_'+str(score[1])+'_'+str(myfilename[cnt]),image)
        # print (myfilename[cnt])
        cnt=cnt+1

if __name__ == "__main__":
    load_image()
    main()