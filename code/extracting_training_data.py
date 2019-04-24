import pydicom
from collections import defaultdict
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

def extract_data(PathToDataset=r'C:\Users\Animesh Garg\Documents\SPIE-AAPM Lung CT Challenge\Training Set',dataset='train'):

    patients = defaultdict(lambda : list())
    
    Patient_IDs = os.listdir(PathToDataset)
    
    for ID in Patient_IDs:
        for dirpath, subdir, filenames in os.walk(os.path.join(PathToDataset,ID)):
            for file in filenames:
                patients[ID].append(os.path.join(dirpath,file))
                
    RefDs = pydicom.read_file(patients[Patient_IDs[6]][10])
    
    IMAGE_SIZE = (RefDs.Rows,RefDs.Columns)
    
    patients_array = defaultdict(lambda :  list())
    data = list()
    
    for ID in Patient_IDs:
        patients_array[ID] = [pydicom.read_file(file) for file in patients[ID]]
        patients_array[ID].sort(key = lambda x: int(x.InstanceNumber))
            
    for index, ID in enumerate(Patient_IDs):
        patient = np.zeros((*IMAGE_SIZE,len(patients_array[ID])) , dtype=np.float32)
        for filenumber, file in enumerate(patients_array[ID]):    
            patient[:,:,filenumber] = file.pixel_array
        data.append(patient)
    
    if dataset=="train":
        pickle.dump(data , open('./data/training_data.p', 'wb'))
    
    if dataset=='test':   
        pickle.dump(data, open('/data/test_data.p','wb'))
#=======================================================================================================================
training_data = pickle.load(open('./data/training_data.p', 'rb'))

training_data[0].max()

RefDs.RescaleIntercept
RefDs.RescaleSlope

sample_image = np.array()
sample_image = training_data[0][:,:,80] + RefDs.RescaleIntercept
plt.hist(sample_image.flatten(),bins=80,color='c')
plt.imshow(sample_image,cmap=plt.cm.gray)