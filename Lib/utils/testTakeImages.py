import os
import time
import datetime
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--emoji', help='nae of the folder', default="")
args = parser.parse_args()
folder_name = args.emoji

assert len(folder_name)>0

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
old_time = time.time()

while True:

    # saving an image every second
    cur_time = time.time()
    if( cur_time-old_time>0.5 ):
        img_name = "opencv_frame_{}.png".format(img_counter)
        path = os.path.join(os.getcwd(),folder_name, img_name)
        cv2.imwrite(path, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        old_time = time.time()

    # updating the canvas
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    # ESC pressed, finish running
    if k%256 == 27:
        print("Escape hit, closing...")
        break



cam.release()

cv2.destroyAllWindows()
