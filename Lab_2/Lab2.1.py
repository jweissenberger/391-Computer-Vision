import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # to see canny edge detection put canny into imshow
    canny = cv2.Canny(frame, 100, 200)

    # to show Harris corner detection, use this code and put frame in imshow
    # orig: 2, 3, 0.02
    dst = cv2.cornerHarris(gray, 2, 3, 0.1)
    # #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    # # Threshold for an optimal value, it may vary depending on the image.
    frame[dst > 0.02 * dst.max()] = [0, 0, 255]

    # Display the resulting frame
    cv2.imshow('Live feed with filter', frame)
    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()