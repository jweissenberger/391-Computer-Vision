import numpy as np
import cv2

if __name__ == '__main__':

    img1 = cv2.imread('sw1.jpg',0)          # queryImage
    img2 = cv2.imread('sw2.jpg',0) # trainImage
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 15 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:15], outImg=None, flags=2)

    cv2.imshow('SIFT keypoint matching', img3)
    key = cv2.waitKey(0)