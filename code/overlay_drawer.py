import cv2

def draw_up_pad(img,alpha,controller_up,img_width,img_height,up_img_width,up_img_height):
    up_posy_1 = int((img_width/2)) - int(up_img_width)
    up_posx_1 = int((img_height/2)) + int(((img_height/2) *0.25))
    up_posy_2 = up_posy_1+up_img_width
    up_posx_2 = up_posx_1+up_img_height
    added_image = cv2.addWeighted(img[up_posx_1:up_posx_2,up_posy_1:up_posy_2,:],alpha,controller_up,1-alpha,0)
    return added_image,up_posx_1,up_posy_1,up_posx_2,up_posy_2

def draw_down_pad(img,alpha,controller_down,img_width,img_height,down_img_width,down_img_height):
    down_posy_1 = int((img_width/2)) - int(down_img_width)
    down_posx_1 = int((img_height/2)) + int(((img_height/2) *0.65))
    down_posy_2 = down_posy_1+down_img_width
    down_posx_2 = down_posx_1+down_img_height
    added_image = cv2.addWeighted(img[down_posx_1:down_posx_2,down_posy_1:down_posy_2,:],alpha,controller_down,1-alpha,0)
    return added_image,down_posx_1,down_posy_1,down_posx_2,down_posy_2    

def draw_right_pad(img,alpha,controller_right,img_width,img_height,right_img_width,right_img_height):
    right_posy_1 = int((img_width/2)) - int(right_img_width) - int((img_width/2) * 0.32)
    right_posx_1 = int((img_height/2)) + int(((img_height/2) *0.35))
    right_posy_2 = right_posy_1+right_img_width
    right_posx_2 = right_posx_1+right_img_height
    added_image = cv2.addWeighted(img[right_posx_1:right_posx_2,right_posy_1:right_posy_2,:],alpha,controller_right,1-alpha,0)
    return added_image,right_posx_1,right_posy_1,right_posx_2,right_posy_2   

def draw_left_pad(img,alpha,controller_left,img_width,img_height,left_img_width,left_img_height):
    left_posy_1 = int((img_width/2)) - int(left_img_width) + int((img_width/2) * 0.25)
    left_posx_1 = int((img_height/2)) + int(((img_height/2) *0.35))
    left_posy_2 = left_posy_1+left_img_width
    left_posx_2 = left_posx_1+left_img_height
    added_image = cv2.addWeighted(img[left_posx_1:left_posx_2,left_posy_1:left_posy_2,:],alpha,controller_left,1-alpha,0)
    return added_image,left_posx_1,left_posy_1,left_posx_2,left_posy_2   

def draw_button(img,alpha,controller_button,img_width,img_height,button_img_width,button_img_height,offset):
    button_posy_1 = int((img_width/2)) - int(button_img_width) + int((img_width/2) * (0.60 )) +offset
    button_posx_1 = int((img_height/2)) + int(((img_height/2) *0.35))
    button_posy_2 = button_posy_1+button_img_width
    button_posx_2 = button_posx_1+button_img_height
    added_image = cv2.addWeighted(img[button_posx_1:button_posx_2,button_posy_1:button_posy_2,:],alpha,controller_button,1-alpha,0)
    return added_image,button_posx_1,button_posy_1,button_posx_2,button_posy_2   