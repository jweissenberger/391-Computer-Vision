"""
Jack Weissenberger
Lab 1, Part 3 Frequency filtering
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

if __name__ == "__main__":

    img = cv2.imread('puppy.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    Y = (np.linspace(-int(gray.shape[0]/2), int(gray.shape[0]/2)-1, gray.shape[0]))
    X = (np.linspace(-int(gray.shape[1]/2), int(gray.shape[1]/2)-1, gray.shape[1]))
    X, Y = np.meshgrid(X, Y)

    # Explore the Butterworth filter
    # U and V are arrays that give all integer coordinates in the 2-D plane
    #  [-m/2 , m/2] x [-n/2 , n/2].
    # Use U and V to create 3-D functions over (U,V)
    U = (np.linspace(-int(gray.shape[0]/2), int(gray.shape[0]/2)-1, gray.shape[0]))
    V = (np.linspace(-int(gray.shape[1]/2), int(gray.shape[1]/2)-1, gray.shape[1]))
    U, V = np.meshgrid(U, V)
    # The function over (U,V) is distance between each point (u,v) to (0,0)
    D = np.sqrt(X*X + Y*Y)
    # create x-points for plotting
    xval = np.linspace(-int(gray.shape[1]/2), int(gray.shape[1]/2)-1, gray.shape[1])
    # Specify a frequency cutoff value as a function of D.max()
    D0 = 0.25 * D.max()

    # The ideal lowpass filter makes all D(u,v) where D(u,v) <= 0 equal to 1
    # and all D(u,v) where D(u,v) > 0 equal to 0
    idealLowPass = D <= D0

    # Filter our small grayscale image with the ideal lowpass filter
    # 1. DFT of image
    print(gray.dtype)
    FTgray = np.fft.fft2(gray.astype(float))
    # 2. Butterworth filter is already defined in Fourier space
    # 3. Elementwise product in Fourier space (notice fftshift of the filter)
    FTgrayFiltered = FTgray * np.fft.fftshift(idealLowPass)
    # 4. Inverse DFT to take filtered image back to the spatial domain
    grayFiltered = np.abs(np.fft.ifft2(FTgrayFiltered))

    # Save the filter and the filtered image (after scaling)
    idealLowPass = ski.img_as_ubyte(idealLowPass / idealLowPass.max())
    grayFiltered = ski.img_as_ubyte(grayFiltered / grayFiltered.max())
    cv2.imwrite("idealLowPass.jpg", idealLowPass)
    cv2.imwrite("grayImageIdealLowpassFiltered.jpg", grayFiltered)

    # Plot the ideal filter and then create and plot Butterworth filters of order
    # n = 1, 2, 3, 4
    plt.plot(xval, idealLowPass[int(idealLowPass.shape[0]/2), :], 'c--', label='ideal')
    colors='brgkmc'
    for n in range(1, 5):
        # Create Butterworth filter of order n
        H = 1.0 / (1 + (np.sqrt(2) - 1)*np.power(D/D0, 2*n))
        # Apply the filter to the grayscaled image
        FTgrayFiltered = FTgray * np.fft.fftshift(H)
        grayFiltered = np.abs(np.fft.ifft2(FTgrayFiltered))
        grayFiltered = ski.img_as_ubyte(grayFiltered / grayFiltered.max())
        cv2.imwrite("grayImageButterworth-n" + str(n) + ".jpg", grayFiltered)
        # cv2.imshow('H', H)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        H = ski.img_as_ubyte(H / H.max())
        cv2.imwrite("butter-n" + str(n) + ".jpg", H)
        # Get a slice through the center of the filter to plot in 2-D
        slice = H[int(H.shape[0]/2), :]
        plt.plot(xval, slice, colors[n-1], label='n='+str(n))
        plt.legend(loc='upper left')

    plt.show()
    plt.savefig('butterworthFilters.jpg', bbox_inches='tight')
