import cv2
from skimage import io
from pathlib import Path
import numpy as np
import argparse
import game_object

from helpers.detector import load_checkpoint, detect_hands, collide_objects
# import overlay_drawer

def main():
    parser = argparse.ArgumentParser(description='Running the hand detection with game overlay')
    parser.add_argument('--fps', type=int, default=1, help='FPS of the video')
    parser.add_argument('--video', type=int, default=0, help='Video source')
    parser.add_argument('--width', type=int, default=640, help='Width of the video')
    parser.add_argument('--height', type=int, default=480, help='Height of the video')
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

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # TODO: Detect hands
        boxes, scores, _, num = detect_hands(frame, graph, session)

        # TODO: Draw hands
        # collision_detection(num, boxes, scores, objects)

        # TODO: Draw game objects overlays
        # overlay_drawer.draw_overlay()

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
