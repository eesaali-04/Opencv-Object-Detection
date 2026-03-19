import cv2

import numpy as np
image = cv2.imread('blobs.jpeg',0)

# initialise paramater setting
params = cv2.SimpleBlobDetector_Params()
# filter by area
params.filterByArea = True
params.minArea = 100