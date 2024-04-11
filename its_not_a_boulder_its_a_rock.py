# NASA COSG CSU Rover Design Team
# Jacob Auman

'''
What is my purpose? ... You look for rocks ... Oh my god:( 
'''
import os
import numpy as np
import pandas as pd
import cv2 
from ultralytics import YOLO


def getImageList():
    # get a list of images from the folder using pandas
    # folder_path = 'test_img'  # Replace with the actual folder path
    folder_path = 'rover_test'  # Replace with the actual folder path
    file_names = os.listdir(folder_path)
    df = pd.DataFrame(file_names, columns=['File Name'])
    image_list = df['File Name'].tolist()
    print(f'Image List: {image_list}')
    return image_list

def displayImage(imgList):
    # choose a random image from the folder
    img_name = np.random.choice(imgList)
    # img_path = f'test_img/{img_name}'
    img_path = f'rover_test/{img_name}'
    img = cv2.imread(img_path)

    height, width, _ = img.shape
    if height != 800:
        scale =  800/height
        img = cv2.resize(img, (int(width * scale), int(height * scale )))

    return img

def mergeImg(img, overlay):
    # overlay the rock_overlay onto the img
    img_annotated = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    return img_annotated

def YoloOBJ(img):
    # find the rock in the image using YOLO object detection
    model = YOLO('yolov8n.pt')
    results = model(img)
    print(results)

def detectRocks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image to grayscale
    edges = cv2.Canny(gray, 100, 200) # apply edge detection algorithm
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # # draw contours on the image
    # img_with_edges = cv2.drawContours(img, contours, -1, (255, 255, 0), 2)
    
    # create a black screen
    black_screen = np.zeros_like(img)
    
    # overlay edge detection on the black screen
    edge_overlay_red = cv2.drawContours(black_screen, contours, -1, (0, 0, 255), 3)
    return edge_overlay_red

def detectRails(img):
    # convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # apply thresholding to create a binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # apply morphological operations to enhance the white regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # find contours of the white regions
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # draw red rectangles around the long strips of white cardboard
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 * h:  # adjust the aspect ratio threshold as needed
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    return img

def main():
    imgList = getImageList()
    img = displayImage(imgList)

    # YoloOBJ(img)
    # rock_overlay = detectRocks(img)
    rock_overlay = detectRails(img)
    # img_annotated = mergeImg(img, rock_overlay)
    # cv2.imshow('Image', img_annotated)
    cv2.imshow('Image', rock_overlay)
    # Discard or Save Image
    key = cv2.waitKey(0)
    num = np.random.randint(0, 1000)
    if key == ord('s'):
        cv2.imwrite(f'annotated_img/annotated_image_{num}.jpg', img_annotated)
    elif key == ord('q'):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()