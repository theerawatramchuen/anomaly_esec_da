## Input Parameter START Here ##
IMAGE_PATH = r'E:/20-9-2022/dev3' # r'E:/@@@temp' # 
CROPPED_PATH = r'E:/20-9-2022/dev3/cropped' # r'E:/@@@temp/cropped' # 

# ROI to cropped position Topleft X,Y and Width, Height
roi_tpleft = (279,195)#(258,157)
roi_w      = 102  #120
roi_h      = 100-2   #116

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
            cv.imwrite(os.path.join(CROPPED_PATH,file),cropped_image)
            # cv.imwrite(os.path.join(CROPPED_PATH,str(time.time())+'.jpg'),cropped_image)
        else:
            print('Template Position out of Range')
        if setup:
            exit()   # run single 1st image

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
