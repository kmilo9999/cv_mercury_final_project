import cv2
from skimage import io
from pathlib import Path
import numpy as np

import overlay_drawer


video_cap = cv2.VideoCapture(0)
video_cap.set(cv2.CAP_PROP_FRAME_WIDTH ,1280)
video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT ,720)

controller_up = cv2.imread("../imgs/up1.png")
controller_down = cv2.imread("../imgs/down.png")
controller_right = cv2.imread("../imgs/right.png")
controller_left = cv2.imread("../imgs/left.png")
controller_button = cv2.imread("../imgs/button.png")

up_height,up_width,channels = controller_up.shape
down_height,down_width,channels = controller_down.shape
right_height,right_width,channels = controller_right.shape
left_height,left_width,channels = controller_left.shape
button_height,button_width,channels = controller_button.shape

print(controller_right.shape)

alpha = 0.01
#PRESS 'ESC' TO CLOSE THE WINDOW 
while cv2.waitKey(1) !=27:
    success, img = video_cap.read()
    img = cv2.flip(img,1)
    img_width,img_height,img_channels = img.shape
    added_image_up,up_posx_1,up_posy_1,up_posx_2,up_posy_2 = overlay_drawer.draw_up_pad(img,alpha,controller_up,img_height,img_width,up_width,up_height)
    img[up_posx_1:up_posx_2,up_posy_1:up_posy_2,:] = added_image_up
    added_image_down,up_posx_1,up_posy_1,up_posx_2,up_posy_2 = overlay_drawer.draw_down_pad(img,alpha,controller_down,img_height,img_width,down_width,down_height)
    img[up_posx_1:up_posx_2,up_posy_1:up_posy_2,:] = added_image_down
    added_image_right,up_posx_1,up_posy_1,up_posx_2,up_posy_2 = overlay_drawer.draw_right_pad(img,alpha,controller_right,img_height,img_width,right_width,right_height)
    img[up_posx_1:up_posx_2,up_posy_1:up_posy_2,:] = added_image_right
    added_image_left,up_posx_1,up_posy_1,up_posx_2,up_posy_2 = overlay_drawer.draw_left_pad(img,alpha,controller_left,img_height,img_width,left_width,left_height)
    img[up_posx_1:up_posx_2,up_posy_1:up_posy_2,:] = added_image_left
    added_image_button,up_posx_1,up_posy_1,up_posx_2,up_posy_2 = overlay_drawer.draw_button(img,alpha,controller_button,img_height,img_width,button_width,button_height,0)
    img[up_posx_1:up_posx_2,up_posy_1:up_posy_2,:] = added_image_button
    added_image_button,up_posx_1,up_posy_1,up_posx_2,up_posy_2 = overlay_drawer.draw_button(img,alpha,controller_button,img_height,img_width,button_width,button_height,button_width +20)
    img[up_posx_1:up_posx_2,up_posy_1:up_posy_2,:] = added_image_button

    cv2.imshow("Image",img)


