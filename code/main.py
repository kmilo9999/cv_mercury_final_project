import cv2
from skimage import io
from pathlib import Path
import numpy as np
from game_object import game_object
import overlay_drawer


video_cap = cv2.VideoCapture(0)
video_cap.set(cv2.CAP_PROP_FRAME_WIDTH ,1280)
video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT ,720)



#list of game objects
scene_objects = []



for i in range(3):
    pos_x = np.random.randint(640)
    pos_y = np.random.randint(360)
    pos = np.array([pos_x,pos_y])
    mosquito = game_object("../imgs/mosquito2.png",pos)
    scene_objects.append(mosquito)


alpha = 0.01
#PRESS 'ESC' TO CLOSE THE WINDOW 
while cv2.waitKey(1) !=27:
    success, screen_buffer = video_cap.read()
    screen_buffer = cv2.flip(screen_buffer,1)
    img_width,img_height,img_channels = screen_buffer.shape
    for i in range(len(scene_objects)):
        scene_objects[i].draw(screen_buffer,alpha)
    

    cv2.imshow("Image",screen_buffer)


