import numpy as np
import pandas as pd
import setting 
import extracting_data
import os
import moviepy.editor as mpy
import helper_functions

P_MALIG = 0.4

def make_gifs(image_path = setting.TRAINING_SET , csv_src_path=os.path.join(setting.DATA_PATH,'test'), color_intensity=1000, color='red'):
    images = extracting_data.load_patient(image_path)
    Patient_IDs = os.listdir(image_path)
    for ID in Patient_IDs:
        images_list = list()
        image = extracting_data.get_pixel_hu(images[ID])
        image , _ = extracting_data.resample(images[ID],image)
        image = helper_functions.normalize_hu(image)
        df = pd.read_csv(os.path.join(csv_src_path,ID+'.csv'))
        prob_map = np.zeros(image.shape)
        for _ , info in df.iterrows():
            coord_z = int(info['nodule_z'])
            coord_y = int(info['nodule_y'])
            coord_x = int(info['nodule_x'])
            prob_map[coord_z-20:coord_z+20,coord_y-10:coord_y+10,coord_x-10:coord_x+10] = float(info["Diagnosis"])*color_intensity
            color_filter = image + prob_map
        for i in range(image.shape[0]):
            images_array = np.zeros((image.shape[2], image.shape[1], 3))
            if color == "blue":
                image0  = image[i,:,:]
                image1 =  image[i,:,:]
                image2 = color_filter[i,:,:]

            elif color == "red":
                image2  = image[i,:,:]
                image1 =  image[i,:,:]
                image0 = color_filter[i,:,:]
    
            elif color == "green":
                image0  = image[i,:,:]
                image2 =  image[i,:,:]
                image1 = color_filter[i,:,:]
            
            images_array[:,:,0] = image0
            images_array[:,:,1] = image1
            images_array[:,:,2] = image2
                
            images_list.append(images_array)
        
        my_clip = mpy.ImageSequenceClip(images_list,fps=10)
        my_clip.write_gif(os.path.join(setting.MAIN_DIRECTORY,ID+'.gif'),fps=10)
