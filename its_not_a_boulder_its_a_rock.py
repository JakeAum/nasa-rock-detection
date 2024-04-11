
import cv2 

while True: 
    # reset score
    left = 0
    right = 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image to grayscale
    edges = cv2.Canny(gray, 100, 200) # apply edge detection algorithm
    
    # take all of the pixels in edges on the left side of the immage and summ them
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]//2):
            left += edges[i][j]
    
    # take all of the pixels in edges on the right side of the immage and summ them
    for i in range(0, img.shape[0]):
        for j in range(img.shape[1]//2, img.shape[1]):
            right += edges[i][j]

    # print the score
    print(left, right)

