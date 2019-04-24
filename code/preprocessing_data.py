import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from sample_generator import sample_creater

dataset = pickle.load(open("../data/training_data.p",'rb'))
Patient_IDs = os.listdir('../Training Set')
Patient_Info = pd.read_csv('../data/final_train.csv')

new_train_set, new_Patient_IDs, nodule_x, nodule_y, nodule_z, diagnosis = sample_creater(dataset, Patient_IDs, Patient_Info)
new_train_set = [train_image -1024 for train_image in new_train_set]

plt.imshow(new_train_set[12][:,:,3],cmap='gray')

new_Patient_IDs = pd.DataFrame(new_Patient_IDs,columns=["Patient_ID"])
nodule_x = pd.DataFrame(nodule_x,columns=["X"])
nodule_y = pd.DataFrame(nodule_y,columns=["Y"])
nodule_z = pd.DataFrame(nodule_z,columns=["Z"])
diagnosis = pd.DataFrame(diagnosis,columns=["diagnosis"])

modified_train_set = pd.concat([new_Patient_IDs,nodule_x,nodule_y,nodule_z,diagnosis],axis=1,ignore_index=False)

modified_train_set.head()
