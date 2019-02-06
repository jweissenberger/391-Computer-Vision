"""
Jack Weissenberger
Lab1 Part 1, Spatial Filtering
"""


import cv2


if __name__ == '__main__':

    """
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

    # gaussian filters on puppy noise data
    img = cv2.imread('noise_puppy.JPG')
