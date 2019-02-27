import numpy as np
import cv2

if __name__ == '__main__':

    img = cv2.imread('LM1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initiate STAR detector
    orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
    kp2 = orb.detect(img)

    img2_kp = cv2.drawKeypoints(img, kp2, None, flags=cv2.DrawMatchesFlags_DEFAULT)


    cv2.imshow('myImage', img2_kp)
    key = cv2.waitKey(0)
