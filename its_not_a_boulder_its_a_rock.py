
import cv2 

while True: 
    # reset score
    left = 0
    right = 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image to grayscale
    edges = cv2.Canny(gray, 100, 200) # apply edge detection algorithm
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # calculate the area of the contours in the left half of the screen
    for contour in contours:
        if cv2.pointPolygonTest(contour, (img.shape[1]//4, img.shape[0]//2), False) == 1:
            left += cv2.contourArea(contour)

    # calculate the area of the contours in the right half of the screen
    for contour in contours:
        if cv2.pointPolygonTest(contour, (3*img.shape[1]//4, img.shape[0]//2), False) == 1:
            right += cv2.contourArea(contour)

    # print the score
    print(left, right)
