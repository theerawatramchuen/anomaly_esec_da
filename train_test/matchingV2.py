## Input Parameter START Here ##
IMAGE_PATH = r'E:/21-9-2022/dev1' # r'E:/@@@temp' # 
CROPPED_PATH = r'E:/21-9-2022/dev1/cropped' # r'E:/@@@temp/cropped' # 

# ROI to cropped position Topleft X,Y and Width, Height
roi_tpleft = (280,194-2)#(258,157)
roi_w      = 100  #120
roi_h      = 100   #116

# Setup roi to run only 1 unit
yes_setup = 0      # 1 is setup ,  0 is run 

## Input Parameter END Here ##

if yes_setup == 0:
   setup = False #
else:
    setup = True

import time
import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import csv
import pandas as pd

data = {
    "file_path":[],
    "top_left_X":[],
    "top_left_Y":[],
    "bottom_right_X":[],
    "bottom_right_Y":[],
    "ver_offset":[],
    "roi_tpleft_X":[],
    "roi_tpleft_Y":[],
    "roi_w":[],
    "roi_h":[],
    "file_training_path":[],
    "training_date_time":[],
    "asm":[]
}

df = pd.DataFrame(data)

roi_bnright = (roi_tpleft[0]+roi_w,roi_tpleft[1]+roi_h)
template = cv.imread('template.jpg',0) # Find Template in image

os.chdir(IMAGE_PATH)
mycounter = 0
for file in os.listdir():
    if file.endswith(".jpg"):
        file_path = f"{IMAGE_PATH}/{file}"
#       start = time.time()
        img = cv.imread(file_path,0)
        img4cropped = img
        img2 = img.copy()
        w, h = template.shape[::-1]
        # method = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 
        #           'cv.TM_CCORR','cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
        img = img2.copy()
        method = eval('cv.TM_CCOEFF')
        res = cv.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
#       end = time.time()

        ver_offset = top_left[1]-433
        if (top_left[0] >= 18) and (top_left[0] <=20) or (top_left[0] >= 5) and (top_left[0] <=6):   # ((age>= 8) and (age<= 12))
            os_roi_tpleft = (roi_tpleft[0],roi_tpleft[1]+ver_offset)
            os_roi_bnright = (roi_bnright[0], roi_bnright[1]+ver_offset)
            cropped_image = img4cropped[os_roi_tpleft[1]:os_roi_bnright[1],os_roi_tpleft[0]:os_roi_bnright[0]]
            #cropped_image = img4cropped[256:381,256:278]
            cv.imwrite(os.path.join(CROPPED_PATH,file),cropped_image)
        else:
            print('Template Position out of Range')
        if setup:
            exit()   # run single 1st image

        df.loc[len(df.index)] = [file_path,top_left[0],top_left[1],bottom_right[0],bottom_right[1]
                                ,ver_offset,roi_tpleft[0],roi_tpleft[1],roi_w,roi_h,np.NaN,np.NaN,np.NaN]
        #print(df.loc[[mycounter]])
        print(mycounter,file_path)
        mycounter = mycounter + 1

    # "file_path":[],
    # "top_left_X":[],
    # "top_left_Y":[],
    # "bottom_right_X":[],
    # "bottom_right_Y":[],
    # "ver_offset":[],
    # "roi_tpleft_X":[],
    # "roi_tpleft_Y":[],
    # "roi_w":[],
    # "roi_h":[],
    # "file_training_path":[],
    # "training_date_time":[],
    # "asm":[]

print(df.describe())
#print(df.info())
df.to_csv('result.csv',index=False)

exit()
###################################################################

