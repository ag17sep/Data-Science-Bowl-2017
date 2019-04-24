import numpy as np
import pandas as pd

def sample_creater(dataset, Patient_IDs, Patient_Info, sample_per_image=5, xdim = 41, ydim = 41, zdim = 21):
    
    sampled_set = []
    diagnosis = []
    
    node_x, node_y, node_z = list(), list(), list()    
    xcut, ycut, zcut = int((xdim-1)/2), int((ydim-1)/2), int((zdim-1)/2)
      
    Patient_IDs = [ID.lower() for ID in Patient_IDs]
    
    sampled_Patient_IDs = []
    sampled_diagnosis = []
    
    for i in range(sample_per_image):
    
        for index, ID in enumerate(Patient_IDs):
            nodule_x = int(Patient_Info["nodule_x"][Patient_Info["Scan Number"] == ID])
            nodule_y = int(Patient_Info["nodule_y"][Patient_Info["Scan Number"] == ID])
            nodule_z = int(Patient_Info["nodule_z"][Patient_Info["Scan Number"] == ID])
            diagnosis = str(Patient_Info["Diagnosis"][Patient_Info["Scan Number"] == ID])
            print(nodule_x, nodule_y, nodule_z,diagnosis)
            xmin = nodule_x - xcut
            ymin = nodule_y - ycut
            zmin = nodule_z - zcut
            xmax = nodule_x + xcut
            ymax = nodule_y + ycut
            zmax = nodule_z + zcut
            #print(xmin,ymin,zmin,xmax,ymax,zmax)
            xrand = np.random.randint(low = xmin,high = xmax,dtype=np.int32)
            yrand = np.random.randint(low = ymin,high = ymax,dtype=np.int32)
            zrand = np.random.randint(low = zmin,high = zmax,dtype=np.int32)
            #print(xrand,yrand,)
            xlow = xrand - xcut if xrand > xcut else 0
            xhigh = xlow + 2*xcut if xlow + 2*xcut < 512 else 512
            ylow = yrand - ycut
            yhigh = yrand + ycut
            zlow = zrand - zcut
            zhigh = zrand + zcut
            
            print(xlow,xhigh,ylow,yhigh,zlow,zhigh)
            new_image = dataset[index][xlow:xhigh, ylow:yhigh, zlow:zhigh]
            
            new_image +=(np.random.rand(xdim-1,ydim-1,zdim-1)*6)
            sampled_set.append(new_image)
            node_x.append(nodule_x - xrand + xcut)
            node_y.append(nodule_y - yrand + ycut)
            node_z.append(nodule_z - zrand + zcut)
            sampled_diagnosis.append(diagnosis)
            sampled_Patient_IDs.append(ID)
    
    return sampled_set, sampled_Patient_IDs, node_x, node_y, node_z, sampled_diagnosis
