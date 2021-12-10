import cv2
import numpy as np

class game_object:
    def __init__(self, texture_path):
        self.texture = cv2.imread(texture_path)
        self.text_width = self.texture.shape[1]
        self.text_height = self.texture.shape[0]
        self.position= np.zeros((1,2))
        self.radius = 10
        self.speed = 0.001
    
    def draw(self, screen_buffer, alpha):
        posy = self.position[1]+self.text_width
        posx = self.position[0]+ self.text_height
        added_image = cv2.addWeighted(screen_buffer[self.position[0]:posx,
                        self.position:posy,:],alpha,self.texture,1-alpha,0)
        screen_buffer[self.position[0]:posx,self.position[1]:posy,:] = added_image

    def move(self, direction):
        self.position = self.position + direction * self.speed

    def get_bounding_sphere(self):
        return self.position,self.radius
        
    