import cv2

import numpy as np
image = cv2.imread('blobs.jpeg',0)

# initialise paramater setting
params = cv2.SimpleBlobDetector_Params()
# filter by area
params.filterByArea = True
params.minArea = 100
# filter by circularity
params.filterByCircularity = True
params.minCircularity = 0.9
# filter by convexity
params.filterByConvexity = True
params.minConvexity = 0.2
# filter by inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
# Create a detector with the perameters
detector = cv2.SimpleBlobDetector_create(params)
# Detect the blobs
keypoints = detector.detect(image)
# Draw the blobs on the image
blank = np.zeros(( (1,1)))
blobs_img = cv2.drawKeypoints(image,keypoints,blank,(0,10,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# display the total number of blobs
number_of_blobs = len(keypoints)
text =  f'Total number of blobs :{number_of_blobs}'
cv2.putText(blobs_img,text,(20,550),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
cv2.imshow('blob detecter',blobs_img)
cv2.waitKey(0)
cv2.destroyAllWindows()