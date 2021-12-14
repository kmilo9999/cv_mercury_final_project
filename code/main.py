import cv2
from skimage import io
from pathlib import Path
import numpy as np
from helpers.game_object import game_object
import argparse
import random

from helpers.detector import load_checkpoint, detect_hands, collide_objects

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
    cap = cv2.VideoCapture(0)
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
        pos_x = np.random.randint(img_width//2)
        pos_y = np.random.randint(img_height//2)
        pos = np.array([pos_x,pos_y])
        direction =  np.array([random.uniform(0, 1),random.uniform(0, 1)])
        mosquito = game_object("../imgs/mosquito2.png",pos,direction)
        scene_objects.append(mosquito)


    alpha = 0.01

    # Read until video is completed
    font = cv2.FONT_HERSHEY_SIMPLEX
    while(cap.isOpened()):
        # Capture frame-by-frame
        _, frame = cap.read()
        
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes, scores = detect_hands(frame, graph, session)

        collide_objects(2, boxes, scores, scene_objects, frame, img_width, img_height)
        
        # Moves the object at random directions with 2D matrix where 1 is right and -1 is left
        for object in scene_objects:
            posx1, posy1, posx2, posy2 = object.get_rectangle()
            _posx1 = int(posx1)
            _posy1 = int(posy1)
            _posx2 = int(posx2) 
            _posy2 = int(posy2)
            strRes = "" + str(img_width) +","+str(img_height)
            strPos ="" +str(_posx1)+","+str(_posy1)+" "+str(_posx2)+","+str(_posy2)
            cv2.putText(frame, 
                strRes+"\\n "+strPos, 
                (50, 50), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
            cv2.rectangle(frame, (int(posx1), int(posy1)), (int(posx2), int(posy2)), (255, 255, 0), 2)
            object.draw(frame,alpha)
       
        
        # Display the resulting frame
        cv2.imshow('Hand Detection', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
