import cv2

import numpy as np
image = cv2.imread('coins.png',0)

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True
params.minCircularity = 0.89
params.filterByConvexity = True
params.minConvexity = 0.2
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(image)

blank = np.zeros(( (1,1)))
coins_img = cv2.drawKeypoints(image,keypoints,blank,(0,10,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_round_coins = len(keypoints)
text =  f'Total number of round coins :{number_of_round_coins}'
cv2.putText(coins_img,text,(20,550),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
cv2.imshow('round coin detecter',coins_img)
cv2.waitKey(0)
cv2.destroyAllWindows()