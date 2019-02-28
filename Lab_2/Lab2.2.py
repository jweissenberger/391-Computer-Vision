import numpy as np
import cv2

if __name__ == '__main__':

    img = cv2.imread('florence-dome.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=10000, scaleFactor=1, scoreType=cv2.ORB_FAST_SCORE)
    kp2 = orb.detect(img)

    img2_kp = cv2.drawKeypoints(img, kp2, None, flags=cv2.DrawMatchesFlags_DEFAULT)


    cv2.imshow('myImage', img2_kp)
    cv2.waitKey(0)
    #cv2.imwrite("increasedScale.jpg", img2_kp)
