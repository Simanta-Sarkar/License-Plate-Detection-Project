import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
import imutils
import easyocr
import csv
from matplotlib import pyplot as plt

def openFile():
    file_path = filedialog.askopenfilename()
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    if len(result) > 0:
        text = result[0][-2]
        save_to_csv(file_path, text)
        print("Number plate:", text)
        show_image_with_text(img, location, text)
    else:
        print("No number plate detected.")

def save_to_csv(file_path, number_plate):
    csv_file = "number_plates.csv"
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([file_path, number_plate])
    print("Number plate saved to", csv_file)

def show_image_with_text(img, location, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text=text, org=(location[0][0][0], location[1][0][1] + 60), fontFace=font, fontScale=1,
                color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

window = Tk()
button = Button(text="Open", command=openFile)
button.pack()
window.mainloop()
