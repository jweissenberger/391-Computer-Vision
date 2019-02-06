"""
Jack Weissenberger
Lab1 Part 1, Spatial Filtering
"""


import cv2


if __name__ == '__main__':

    img = cv2.imread('puppy.jpg')

    print("Here is the original image")
    cv2.imshow('myImage', img)
    key = cv2.waitKey(0)  # closes the image when any button is pressed

    k = int(input("Enter an integer to be the value of k: "))
    print("Here is the blurred image")
    boxed = cv2.boxFilter(img, -1, ksize=(k, k))
    cv2.imshow('myImage', boxed)
    key = cv2.waitKey(0)
    cv2.imwrite('blured_puppy.JPG',boxed)

    print("\nThis is what the filter would look like as a matrix")
    for i in range(k):
        for j in range(k):
            print('1/%d, ' % k, end='')
        print("")

    """
    # 3.2.1

    # gaussian filters on puppy noise data
    img = cv2.imread('noise_puppy.JPG')
    cv2.imshow('myImage', img)

    # gaussian blur
    key = cv2.waitKey(0)
    gauss = cv2.GaussianBlur(img, (27, 27), 0)
    cv2.imwrite('gauss_puppy.JPG', gauss)
    # the larger the kernel of the blur the mor of the noise seems to be reduced but
    # the image also becomes more blurred and we seem to lose more information in the photo
    cv2.imshow("gaus", gauss)
    key = cv2.waitKey(0)

    # this is the median blur
    median = cv2.medianBlur(img, 27)
    cv2.imwrite('median_puppy.JPG', median)
    cv2.imshow("med", median)
    key = cv2.waitKey(0)
    
    # 3.2.2
    img = cv2.imread('noise_puppy.jpg')
    edges = cv2.Canny(img,100,200)
    cv2.imshow("puppy edges", edges)
    key = cv2.waitKey(0)
    img = cv2.imread('window-05-01.jpg')
    edges2 = cv2.Canny(img,100,200)
    cv2.imshow("window edges", edges2)
    key = cv2.waitKey(0)
    """
