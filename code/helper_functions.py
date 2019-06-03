import numpy as np
import pandas as pd
import cv2
import random

CUBE_SIZE = 32

def normalize_image(image):
    MAX_SIZE = image.max()
    MIN_SIZE = image.min()  
    image -=MIN_SIZE
    image = image.astype(np.float)
    image /= (MAX_SIZE - MIN_SIZE)
    image *= 255
    image = image.astype(np.uint8)
    return image

def normalize_hu(image):
    MIN_BOUND = -1024
    MAX_BOUND = 400
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    image *= 255. 
    return image
    
def save_image(target_path,image,rows,cols):
    assert rows * cols == image.shape[0]
    img_width = image.shape[1]
    img_height = image.shape[1]
    res = np.zeros((rows*img_height,cols*img_width) , dtype = np.uint8)
    for row in range(rows):
        for col in range(cols):
            target_y = row*img_height
            target_x = col*img_width
            res[target_y:target_y+img_height,target_x:target_x+img_width] = image[row*col + col]
    
    cv2.imwrite(target_path+'.jpg',res)
    
def load_image(src_path,rows,cols,size):
    image = cv2.imread(src_path,cv2.IMREAD_GRAYSCALE)
    res = np.zeros((rows*cols,size,size))    
    img_height = size
    img_width = size
    for row in range(rows):
        for col in range(cols):
            src_y = row * img_height
            src_x = col * img_width
            res[row * cols + col] = image[src_y:src_y + img_height, src_x:src_x + img_width] 
    return res

def get_pos_cube_from_image(img_3d, center_x, center_y, center_z, block_size):
    start_x = max((center_x - block_size//2),0)
    if start_x + block_size//2 > img_3d.shape[2]:
        start_x = img_3d.shape[2] - block_size
    start_y = max((center_y - block_size//2),0)
    if start_y + block_size//2 > img_3d.shape[1]:
       start_y = img_3d.shape[1] - block_size
    start_z = max((center_z - block_size//2),0)
    if start_z + block_size//2 > img_3d.shape[0]:
        start_z = img_3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    res = img_3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return res

def get_neg_cube_from_image(img_3d, center_x, center_y, center_z, block_size):
    start_x = random.randint(0,int(center_x - block_size)) if center_x > block_size else random.randint(int(center_x+1),int(img_3d.shape[2]-block_size))
    start_y = random.randint(0,int(center_y - block_size)) if center_y > block_size else random.randint(int(center_y+1),int(img_3d.shape[1]-block_size))
    start_z = random.randint(0,int(center_z - block_size)) if center_z > block_size else random.randint(int(center_z+1),int(img_3d.shape[0]-block_size))
    start_x = int(start_x)
    start_y = int(start_y)
    start_z = int(start_z)
    res = img_3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return res

        
          
class XYRange:
    def __init__(self, x_min, x_max, y_min, y_max, chance=1.0):
        self.chance = chance
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.last_x = 0
        self.last_y = 0
    
    def get_last_xy_txt(self):
        res = "x_" + str(int(self.last_x * 100)).replace("-", "m") + "-" + "y_" + str(int(self.last_y * 100)).replace("-", "m")
        return res

def random_scale_img(img, xy_range, lock_xy=False):
    if random.random() > xy_range.chance:
        return img
    
    scale_x = random.randint(xy_range.x_min,xy_range.x_max)
    scale_y = random.randint(xy_range.y_min,xy_range.y_max)
    if lock_xy:
        scale_y = scale_x
    
    org_height, org_width = img.shape[1:3]
    xy_range.last_x = scale_x
    xy_range.last_y = scale_y
    
    res = np.zeros(img.shape)
    for i in range(img.shape[0]):
        scaled_width = int(org_width * scale_x)
        scaled_height = int(org_height * scale_y)
        scaled_img = cv2.resize(img[i], (scaled_width,scaled_height), interpolation=cv2.INTER_CUBIC)
        if scaled_width < org_width:
            extend_left = (org_width - scaled_width) / 2
            extend_right = org_width - scaled_width - extend_left
            scaled_img = cv2.copyMakeBorder(scaled_img, 0, 0, extend_left, extend_right, borderType=cv2.BORDER_CONSTANT)
            scaled_width = org_width
        
        if scaled_height < org_height:
            extend_top = (org_height - scaled_height) / 2
            extend_bottom = org_height - extend_top - scaled_height
            scaled_img = cv2.copyMakeBorder(scaled_img, extend_top, extend_bottom, 0, 0,  borderType=cv2.BORDER_CONSTANT)
            scaled_height = org_height

        start_x = int((scaled_width - org_width) / 2)
        start_y = int((scaled_height - org_height) / 2)
        res[i] = scaled_img[start_y: start_y + org_height, start_x: start_x + org_width]
    
    return res

def random_translate_img(img, xy_range):
    if random.random() > xy_range.chance:
        return img
    
    org_height, org_width = img.shape[1:3]
    translate_x = random.randint(xy_range.x_min, xy_range.x_max)
    translate_y = random.randint(xy_range.y_min, xy_range.y_max)
    trans_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])

    border_const = cv2.BORDER_CONSTANT
    res = np.zeros(img.shape)
    for i in range(img.shape[0]):
        res[i,:,:] = cv2.warpAffine(img[i,:,:], trans_matrix, (org_width, org_height), borderMode=border_const)
    xy_range.last_x = translate_x
    xy_range.last_y = translate_y
    return res

def random_rotate_img(img, chance, min_angle, max_angle):
    if random.random() > chance:
        return img

    angle = random.randint(min_angle, max_angle)
    center = (img.shape[1] / 2, img.shape[2] / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    res = np.zeros(img.shape)
    for i in range(img.shape[0]):
        res[i,:,:] = cv2.warpAffine(img[i,:,:], rot_matrix, dsize=img.shape[1:3], borderMode=cv2.BORDER_CONSTANT)
    return res

def random_flip_img(img, horizontal_chance=0, vertical_chance=0):
    flip_horizontal = False
    if random.random() < horizontal_chance:
        flip_horizontal = True

    flip_vertical = False
    if random.random() < vertical_chance:
        flip_vertical = True

    if not flip_horizontal and not flip_vertical:
        return img

    flip_val = 1
    if flip_vertical:
        flip_val = -1 if flip_horizontal else 0

    res = np.zeros(img.shape)
    for i in range(img.shape[0]):
        res[i,:,:] = cv2.flip(img[i,:,:], flip_val)
    return res
                

