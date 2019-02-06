"""
Jack Weissenberger
Lab 1 part 2, Frequency Analysis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski


if __name__ == '__main__':

    img = cv2.imread('puppy.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Gray image', gray)
    # key = cv2.waitKey(0)

    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
    # create the figure
    fig = plt.figure()
    ax = plt.figure().gca(projection='3d')
    ax.plot_surface(xx, yy, gray, rstride=1, cstride=1, linewidth=0)
    plt.savefig('3-D_fourier_coefficients.jpg')
    plt.show()

    F2_gray = np.fft.fft2(gray)
    # Plot the magnitude and the log(magnitude + 1) as images (view from the top)
    magnitudeImage = np.fft.fftshift(np.abs(F2_gray))
    magnitudeImage = magnitudeImage / magnitudeImage.max()   # scale to [0, 1]
    magnitudeImage = ski.img_as_ubyte(magnitudeImage)
    cv2.imshow('Magnitude plot', magnitudeImage)
    cv2.imwrite('Two_dim_magnitude.JPG', magnitudeImage)

    # log magnitude
    logMagnitudeImage = np.fft.fftshift(np.log(np.abs(F2_gray)+1))
    logMagnitudeImage = logMagnitudeImage / logMagnitudeImage.max()   # scale to [0, 1]
    logMagnitudeImage = ski.img_as_ubyte(logMagnitudeImage)
    cv2.imshow('Log Magnitude plot', logMagnitudeImage)
    cv2.imwrite('Two_dim_log_magnitude.JPG', logMagnitudeImage)
    cv2.waitKey(0)
