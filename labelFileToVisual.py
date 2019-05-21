import numpy as np
import cv2
import os


dataDir = 'D:dataset\labels\\'
dataOut = 'D:labels2\\'
for filename in os.listdir(dataDir):
    image = cv2.imread(dataDir+filename)
    image2 = np.zeros_like(image)
    s = np.argwhere(image>0)
    # print(s.shape)
    for i in range(len(s)):
        image2[s[i,0],s[i,1], :] =[250,250,250]
    cv2.imwrite(dataOut+filename,image2)