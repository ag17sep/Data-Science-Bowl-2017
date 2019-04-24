import pydicom
import os
import pandas as pd

def extract_metadata(PathToDataset=r"C:\Users\Animesh Garg\Documents\SPIE-AAPM Lung CT Challenge\Training Set",dataset="train"):

    IDs = list()
    Sexes = list()
    Ages = list()
    
    IDs = os.listdir(PathToDataset)
    
    for ID in IDs:
        for dirpath, dirnames, filenames in os.walk(os.path.join(PathToDataset,ID)):
            if (len(filenames) > 0) and (".dcm" in filenames[0].lower()):
                ds = pydicom.read_file(os.path.join(dirpath,filenames[0]))
                Sexes.append(ds.PatientSex)
                Ages.append(int(ds.PatientAge[1:3]))
        
    IDs = pd.DataFrame(IDs,columns=['Scan Number'])
    Ages = pd.DataFrame(Ages,columns=['Patient Age'])
    Sexes = pd.DataFrame(Sexes,columns=['Patient Sex'])
    patient_info = pd.concat([IDs,Ages,Sexes],axis=1,ignore_index=False)
    patient_info["Scan Number"] = patient_info["Scan Number"].apply(str.lower)
    
    if dataset=="train" :
        train_data = pd.read_excel("../data/CalibrationSet_NoduleData.xlsx")
        train_data["Scan Number"] = train_data["Scan Number"].apply(str.lower)
        final_train = pd.merge(patient_info,train_data,on=["Scan Number"])
        
        final_train.to_csv("../data/final_train.csv",index=False)
    
    if dataset == 'test':
        test_data = pd.read_excel("../data/TestSet_NoduleData.xlsx")
        test_data["Scan Number"] = test_data["Scan Number"].apply(str.lower)
        final_test = pd.merge(patient_info,test_data,on=['Scan Number'])
        final_test.to_csv("../data/final_test.csv",index=False)
        