import cv2
from skimage import io
from pathlib import Path
import numpy as np
from game_object import game_object
import overlay_drawer
import argparse
import random

from helpers.detector import load_checkpoint, detect_hands, collide_objects

# video_cap = cv2.VideoCapture(0)
# video_cap.set(cv2.CAP_PROP_FRAME_WIDTH ,1280)
# video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT ,720)



#list of game objects
#scene_objects = []




#PRESS 'ESC' TO CLOSE THE WINDOW 
# while cv2.waitKey(1) !=27:
#     success, screen_buffer = video_cap.read()
#     screen_buffer = cv2.flip(screen_buffer,1)
#     img_width,img_height,img_channels = screen_buffer.shape
#     for i in range(len(scene_objects)):
#         scene_objects[i].draw(screen_buffer,alpha)
    

#     cv2.imshow("Image",screen_buffer)

# import overlay_drawer

def main():
    parser = argparse.ArgumentParser(description='Running the hand detection with game overlay')
    parser.add_argument('--fps', type=int, default=1, help='FPS of the video')
    parser.add_argument('--video', type=int, default=0, help='Video source')
    parser.add_argument('--width', type=int, default=1280, help='Width of the video')
    parser.add_argument('--height', type=int, default=720, help='Height of the video')
    args = parser.parse_args()

    #list of game objects
    scene_objects = []

    

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("Video resolution: {}x{}".format(img_width, img_height))
    print("FPS: {}".format(fps))

    # Name window
    cv2.namedWindow('Hand Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hand Detection', img_width, img_height)
    graph, session = load_checkpoint()

    #init scene objects
    for i in range(3):
        pos_x = np.random.randint(img_width)
        pos_y = np.random.randint(img_height)
        pos = np.array([pos_x,pos_y])
        direction =  np.array([random.uniform(0, 1),random.uniform(0, 1)])
        mosquito = game_object("../imgs/mosquito2.png",pos,direction)
        scene_objects.append(mosquito)


    alpha = 0.01


    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        _, frame = cap.read()
        
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #boxes, scores, _, num = detect_hands(frame, graph, session)

        #scene_objects = collide_objects(num, boxes, scores, scene_objects, frame)

        # Moves the object at random directions with 2D matrix where 1 is right and -1 is left
        for object in scene_objects:
            # TODO: Initialize position to move the object
            object.move()
            # TODO: Pass in arguments into the draw method
            object.draw(frame,alpha)

        # Display the resulting frame
        cv2.imshow('Hand Detection', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

# video_cap = cv2.VideoCapture(0)
# video_cap.set(cv2.CAP_PROP_FRAME_WIDTH ,1280)
# video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT ,720)

# controller_up = cv2.imread("../imgs/up1.png")
# controller_down = cv2.imread("../imgs/down.png")
# controller_right = cv2.imread("../imgs/right.png")
# controller_left = cv2.imread("../imgs/left.png")
# controller_button = cv2.imread("../imgs/button.png")

# up_height,up_width,channels = controller_up.shape
# down_height,down_width,channels = controller_down.shape
# right_height,right_width,channels = controller_right.shape
# left_height,left_width,channels = controller_left.shape
# button_height,button_width,channels = controller_button.shape

# print(controller_right.shape)

# alpha = 0.01
# #PRESS 'ESC' TO CLOSE THE WINDOW 
# while cv2.waitKey(1) !=27:
#     success, img = video_cap.read()
#     img = cv2.flip(img,1)
#     img_width,img_height,img_channels = img.shape
#     added_image_up,up_posx_1,up_posy_1,up_posx_2,up_posy_2 = overlay_drawer.draw_up_pad(img,alpha,controller_up,img_height,img_width,up_width,up_height)
#     img[up_posx_1:up_posx_2,up_posy_1:up_posy_2,:] = added_image_up
#     added_image_down,up_posx_1,up_posy_1,up_posx_2,up_posy_2 = overlay_drawer.draw_down_pad(img,alpha,controller_down,img_height,img_width,down_width,down_height)
#     img[up_posx_1:up_posx_2,up_posy_1:up_posy_2,:] = added_image_down
#     added_image_right,up_posx_1,up_posy_1,up_posx_2,up_posy_2 = overlay_drawer.draw_right_pad(img,alpha,controller_right,img_height,img_width,right_width,right_height)
#     img[up_posx_1:up_posx_2,up_posy_1:up_posy_2,:] = added_image_right
#     added_image_left,up_posx_1,up_posy_1,up_posx_2,up_posy_2 = overlay_drawer.draw_left_pad(img,alpha,controller_left,img_height,img_width,left_width,left_height)
#     img[up_posx_1:up_posx_2,up_posy_1:up_posy_2,:] = added_image_left
#     added_image_button,up_posx_1,up_posy_1,up_posx_2,up_posy_2 = overlay_drawer.draw_button(img,alpha,controller_button,img_height,img_width,button_width,button_height,0)
#     img[up_posx_1:up_posx_2,up_posy_1:up_posy_2,:] = added_image_button
#     added_image_button,up_posx_1,up_posy_1,up_posx_2,up_posy_2 = overlay_drawer.draw_button(img,alpha,controller_button,img_height,img_width,button_width,button_height,button_width +20)
#     img[up_posx_1:up_posx_2,up_posy_1:up_posy_2,:] = added_image_button

#     cv2.imshow("Image",img)
