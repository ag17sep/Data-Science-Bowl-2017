import pydicom
from collections import defaultdict
import os
import cv2
import numpy as np
import scipy.misc
import setting
import helper_functions
import pandas as pd
import scipy

CUBE_SIZE = 32

def load_patient(PathToDataset=setting.TRAINING_SET):

    patients = defaultdict(lambda : list())   
    Patient_IDs = os.listdir(PathToDataset)
    for ID in Patient_IDs:
        for dirpath, subdir, filenames in os.walk(os.path.join(PathToDataset,ID)):
            for file in filenames:
                patients[ID].append(os.path.join(dirpath,file))
    
    Patients_array = defaultdict(lambda :  list())   
    for ID in Patient_IDs:
        Patients_array[ID] = [pydicom.read_file(file) for file in patients[ID]]
        Patients_array[ID].sort(key = lambda x: int(x.InstanceNumber))
        
    return Patients_array, Patient_IDs


def get_pixel_hu(patient_array):

    images = np.stack([slices.pixel_array for slices in patient_array])
    images = images.astype(np.int16)
    for section in range(len(patient_array)):
        slope = patient_array[section].RescaleSlope
        intercept = patient_array[section].RescaleIntercept
        if slope != 1:
            images = slope * images.astype(np.float32)
            images = images.astype(np.int16)
        images[section] = images[section] + np.int16(intercept)
    return np.array(images, dtype = np.int16)

def resample(patient_array,image,new_spacing=[1,1,1]):
    
    spacing = map(float, [patient_array[0].SliceThickness] + list(patient_array[0].PixelSpacing))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_shape = image.shape * resize_factor
    real_new_shape = np.round(new_shape)
    real_resize_factor = real_new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    new_image = scipy.ndimage.interpolation.zoom(image,real_resize_factor)
    return new_image, real_resize_factor

def generate_data(PathToDataset=setting.TRAINING_SET,dataset='Train'):   
    Patient_Array, _ = load_patient(PathToDataset)
    Patient_Info = pd.read_csv(os.path.join(setting.DATA_PATH,"Final_"+dataset+".csv"))
    
    for index, patient in Patient_Info.iterrows():
        if os.path.exists(os.path.join(setting.MAIN_DIRECTORY,dataset,str(patient["Scan Number"]))):
            continue   
        
        os.makedirs(os.path.join(setting.MAIN_DIRECTORY,dataset,str(patient["Scan Number"])))
        coord_x = patient["nodule_x"]
        coord_y = patient["nodule_y"]
        coord_z = patient["nodule_z"]
        
        position = np.array([coord_z,coord_y,coord_x],dtype=np.float32)
        image = get_pixel_hu(Patient_Array[str(patient["Scan Number"])])
        image, resize_factor = resample(Patient_Array[str(patient["Scan Number"])], image)
        image = helper_functions.normalize_hu(image)
        new_position = position * resize_factor
        new_position = np.around(new_position).astype(np.int16)
        
        sampled_pos_image = helper_functions.get_pos_cube_from_image(image, new_position[2], new_position[1], new_position[0], CUBE_SIZE)  
        helper_functions.save_image(os.path.join(setting.MAIN_DIRECTORY,dataset,patient["Scan Number"],"pos"),sampled_pos_image,4,8)
        
        diagnosis = 0 if "benign" in patient['Diagnosis'].lower() else 1
        patient_data = [CUBE_SIZE//2, CUBE_SIZE//2, CUBE_SIZE//2, 1, diagnosis]
        patient_info = pd.DataFrame(columns=['nodule_x','nodule_y','nodule_z','nod_present','diagnosis'])
        patient_info.loc[0] = patient_data
        
        sampled_neg_image = helper_functions.get_neg_cube_from_image(image, new_position[2], new_position[1], new_position[0], CUBE_SIZE)
        helper_functions.save_image(os.path.join(setting.MAIN_DIRECTORY,dataset,patient["Scan Number"],"neg"),sampled_neg_image,4,8)
        
        patient_info.to_csv((os.path.join(setting.MAIN_DIRECTORY,dataset,patient["Scan Number"],patient["Scan Number"]))+'.csv',index=False)
        
