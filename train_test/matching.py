# Loop for all images in IMAGE_PATH folder
IMAGE_PATH = 'E:/7-9-2022/dev1'  # E:\2-9-2022\black-not\error_images\temp
CROPPED_PATH = 'E:/7-9-2022/dev1/cropped'

# ROI to cropped position Topleft X,Y and Width, Height
roi_tpleft = (270,203-12)
roi_w = 97
roi_h = 86

import time
import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import csv

roi_bnright = (roi_tpleft[0]+roi_w,roi_tpleft[1]+roi_h)
template = cv.imread('template.jpg',0) # Find Template in image

os.chdir(IMAGE_PATH)
myf = open('result.csv', 'w',newline='') 
writer = csv.writer(myf)
row = ["File Path",'Top-left','Bottom-Right','Time']
writer.writerow(row)
# iterate through all file
mycounter = 0
for file in os.listdir():
    mycounter = mycounter + 1
    # Check whether file is in text format or not
    if file.endswith(".jpg"):
        file_path = f"{IMAGE_PATH}\{file}"
        start = time.time()

        # print (file_path)
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

        end = time.time()
        row = [file_path,top_left,bottom_right,time.time()]
        writer.writerow(row)
        ver_offset = top_left[1]-433
        if (top_left[0] >= 18) and (top_left[0] <=20) or (top_left[0] >= 5) and (top_left[0] <=6):   # ((age>= 8) and (age<= 12))
            os_roi_tpleft = (roi_tpleft[0],roi_tpleft[1]+ver_offset)
            os_roi_bnright = (roi_bnright[0], roi_bnright[1]+ver_offset)
            print (mycounter,file_path,top_left[0],top_left[1],ver_offset,
                roi_tpleft[0],roi_tpleft[1],os_roi_tpleft,os_roi_bnright)#bottom_right[0],bottom_right[1],time.time(),'Sec')
            cropped_image = img4cropped[os_roi_tpleft[1]:os_roi_bnright[1],os_roi_tpleft[0]:os_roi_bnright[0]]
            #cropped_image = img4cropped[256:381,256:278]
            cv.imwrite(os.path.join(CROPPED_PATH,str(time.time())+'.jpg'),cropped_image)
 
        #exit()   # Unomment this line to run single 1st image

myf.close()    

print (mycounter)
exit()

cv.imwrite(os.path.join(CROPPED_PATH,str(time.time())+'.jpg'),cropped_image)

img = cv.imread('test1.jpg',0)
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
cv.rectangle(img,top_left, bottom_right, 255, 2)
#plt.subplot(121),plt.imshow(res,cmap = 'gray')
#plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.plot(122),plt.imshow(img,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.show()


exit()
