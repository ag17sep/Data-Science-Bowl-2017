import numpy as np
import pandas as pd
import os
import helper_functions
import setting
import neural_network 
import extracting_data

STEP = 20
CUBE_SIZE = 32
P_TH = 0.3

def prepare_image_for_net3D(image, mean_value=None):
    image = image.astype(np.float32)
    max_value = image.max()
    if mean_value is not None:
        image -= mean_value
    image /= max_value
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2], 1)
    return image

def predict_nodules(images_path, load_weight_path=os.path.join(setting.MODEL_WEIGHTS,'model_luna16_full__fs_best.hd5') , csv_target_path = setting.CSV_TARGET):
    test_images = extracting_data.load_patient(images_path)
    Patient_IDs = os.listdir(images_path)
    model = neural_network.neural_network(input_shape=(CUBE_SIZE,CUBE_SIZE,CUBE_SIZE,1), load_weight_path = load_weight_path)
    for ID in Patient_IDs:    
        image = test_images[ID]
        image = extracting_data.get_pixel_hu(image)
        image, _ = extracting_data.resample(test_images[ID],image)
        image = helper_functions.normalize_hu(image)
        predict_vol_shape_list = [0, 0, 0]
        for dim in range(3):
            dim_indent = 0
            while dim_indent + CUBE_SIZE < image.shape[dim]:
                predict_vol_shape_list[dim] += 1
                dim_indent +=STEP
        
        predict_vol_shape = np.array(predict_vol_shape_list)
        predict_vol = np.zeros(predict_vol_shape,dtype=np.float)
        batch_size = 5 
        patient_predictions_csv = []
        batch_list = []
        batch_list_coord = []
        for z in range(predict_vol_shape[0]):
            for y in range(predict_vol_shape[1]):
                for x in range(predict_vol_shape[2]):
                    cube_img = image[z*STEP:z*STEP + CUBE_SIZE,y*STEP:y*STEP + CUBE_SIZE,x*STEP:x*STEP + CUBE_SIZE]
                    img_prep = prepare_image_for_net3D(cube_img)
                    batch_list.append(img_prep)
                    batch_list_coord.append((z,y,x))
                    if (len(batch_list)%batch_size) == 0:
                        batch_data = np.vstack(batch_list)
                        p = model.predict(batch_data,batch_size = batch_size)
                        for i in range(len(p[0])):
                            p_z = batch_list_coord[i][0]
                            p_y = batch_list_coord[i][1]
                            p_x = batch_list_coord[i][2]
                            nodule_chance = p[i][0]
                            predict_vol[p_z, p_y, p_x] = nodule_chance
                            if nodule_chance > P_TH:
                                p_z = np.round(p_z * STEP + CUBE_SIZE/2)
                                p_y = np.round(p_y * STEP + CUBE_SIZE/2)
                                p_x = np.round(p_x * STEP + CUBE_SIZE/2)
                                patient_predictions_csv_line = [ID, p_z, p_y, p_x, p[i,1]]
                                patient_predictions_csv.append(patient_predictions_csv_line)
                        
                        batch_list = []
                        batch_list_coord = []
        
        df = pd.DataFrame(patient_predictions_csv,columns=['Scan Number','nodule_z','nodule_y','nodule_x',"Diagnosis"])
        if not os.path.exists(os.path.join(csv_target_path,ID+".csv")):
            df.to_csv(os.path.join(csv_target_path,ID+'.csv'),index=False)
        else:
            orig = pd.read_csv(os.path.join(csv_target_path,ID+'.csv'))
            orig.append(df)
        