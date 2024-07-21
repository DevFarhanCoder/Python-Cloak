import cv2
import numpy as np
import time

# Check OpenCV version
print(cv2.__version__)

# Capture video from the webcam
capture_video = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not capture_video.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Give the camera some time to warm up
time.sleep(1)

# Capture the background (assuming the background is static and has no red object)
background = None
for i in range(60):
    ret, background = capture_video.read()
    if not ret:
        continue

background = np.flip(background, axis=1)

# Main loop to read from the video feed
while capture_video.isOpened():
    ret, img = capture_video.read()
    if not ret:
        break

    img = np.flip(img, axis=1)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the HSV range for detecting red color
    lower_red1 = np.array([0, 120, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask1 = mask1 + mask2

    # Refine the mask
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

    mask2 = cv2.bitwise_not(mask1)

    # Generate the final output
    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(img, img, mask=mask2)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow("INVISIBLE MAN", final_output)

    # Break the loop if 'ESC' is pressed
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release the webcam and close all OpenCV windows
capture_video.release()
cv2.destroyAllWindows()
