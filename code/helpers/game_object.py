import cv2
import numpy as np

class game_object:
    def __init__(self, texture_path,position,direction):
        self.texture = cv2.imread(texture_path)
        self.text_width = int(self.texture.shape[1])
        self.text_height = int(self.texture.shape[0])
        self.position = position
        self.radius = 10
        self.speed = 0.001
        self.direction =direction
    
    def draw(self, screen_buffer, alpha):
        posy2 = int(self.position[1]+self.text_width)
        posx2 = int(self.position[0]+ self.text_height)
        posx1 = int(self.position[0])
        posy1 = int(self.position[1])
        added_image = cv2.addWeighted(screen_buffer[posx1:posx2,posy1:posy2,:],alpha,self.texture,1-alpha,0)
        screen_buffer[posx1:posx2,posy1:posy2,:] = added_image

    def move(self, direction = None):
        if direction is not None:
            self.direction = direction
        self.position = self.position + self.direction * self.speed

    def get_rectangle(self):
        top = int(self.position[0])
        left = int(self.position[1])
        right = int(self.position[1]+self.text_width)
        bott = int(self.position[0]+ self.text_height)
        return left, top, right, bott
