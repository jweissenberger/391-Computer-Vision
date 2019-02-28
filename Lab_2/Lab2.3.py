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
    print(len(matches))
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], outImg=None, flags=2)

    cv2.imshow('SIFT keypoint matching', img3)
    key = cv2.waitKey(0)
    """

    img1 = cv2.imread('sw1.jpg')          # queryImage
    img2 = cv2.imread('sw2.jpg')

    # harris corner
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    # harris corners
    dst1 = cv2.cornerHarris(gray1, 2, 3, 0.04)
    # #result is dilated for marking the corners, not important
    dst1 = cv2.dilate(dst1, None)
    # harris corners
    dst2 = cv2.cornerHarris(gray2, 2, 3, 0.02)
    # #result is dilated for marking the corners, not important
    dst2 = cv2.dilate(dst2, None)

    # # Threshold for an optimal value, it may vary depending on the image.
    img1[dst1 > 0.02 * dst1.max()] = [0, 0, 255]
    img2[dst2 > 0.02 * dst2.max()] = [0, 0, 255]

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    orb = cv2.ORB_create()
    kp1, des1 = orb.compute(img1, dst1)
    kp2, des2 = orb.compute(img2, dst2)

    matches2 = bf.match(des1, des2)
    matches2 = sorted(matches2, key = lambda x:x.distance)

    img4 = cv2.drawMatchesKnn(img1, img2, gray2, dst2)
    #img4 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], outImg=None, flags=2)

    cv2.imshow('Harris Corners keypoint matching', img4)
    key = cv2.waitKey(0)
    """