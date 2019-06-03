import numpy as np
import pandas as pd
import os
import random
from keras.optimizers import Adam, SGD
from keras.layers import Input, Conv3D, MaxPooling3D, LeakyReLU, Flatten, Dense, Dropout, AveragePooling3D, concatenate
from keras.models import Model
from keras.metrics import binary_crossentropy, mean_absolute_error
from keras import backend as K
import helper_functions
import setting

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

CUBE_SIZE = 32

def prepare_image_for_net3D(image, mean_value=None):
    image = image.astype(np.float32)
    if mean_value is not None:
        image -= mean_value
    image /= 255.
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2], 1)
    return image

class CustomLoss(object):
    
    def __init__(self):
        
        self.__name__ = 'custom_loss' 
    def obj_loss(self, y_true, y_pred):
         loss_obj = K.binary_crossentropy(y_true[...,0],y_pred[...,0]) #* indicator_o
         return loss_obj
    
    def class_loss(self, y_true, y_pred):
        loss_class = K.binary_crossentropy(y_true[...,1],y_pred[...,1]) #* indicator_class
        return loss_class
    
    def l_obj(self, y_true, y_pred):
        return self.obj_loss(y_true, y_pred)

    def l_class(self, y_true, y_pred):
        return self.class_loss(y_true, y_pred)
    
    def __call__(self, y_true, y_pred):
        
        total_obj_loss = self.obj_loss(y_true, y_pred)
        total_class_loss = self.class_loss(y_true, y_pred)
        
        loss = total_obj_loss + total_class_loss 
        return loss

def data_generator(src_path,batch_size):
    img_list = list()
    node_predict = list()
    out_class = list()
    
    xy_range_scale = helper_functions.XYRange(1,5,1,5,0.5)
    
    Patient_IDs = os.listdir(src_path)
    for ID in Patient_IDs:
        for dirpath, _, filenames in os.walk(os.path.join(src_path,ID)):
            for file in filenames:
                if "pos" in file.lower():
                    pos_image = helper_functions.load_image(os.path.join(dirpath,file),4,8,CUBE_SIZE)
                elif "neg" in file.lower():
                    neg_image = helper_functions.load_image(os.path.join(dirpath,file),4,8,CUBE_SIZE)
                else:
                    info = pd.read_csv(os.path.join(dirpath,file))
        scaled_pos = helper_functions.random_scale_img(pos_image, xy_range_scale, lock_xy=False)
        scaled_pos = prepare_image_for_net3D(scaled_pos)
        img_list.append(scaled_pos)
        node_predict.append(1)       
        out_class.append(int(info["diagnosis"]))
        #out_malignancy.append((CUBE_SIZE//2,CUBE_SIZE//2,CUBE_SIZE//2))
        
        xy_range_translate = helper_functions.XYRange(-CUBE_SIZE/2,CUBE_SIZE/2,-CUBE_SIZE/2,CUBE_SIZE/2,0.5)
        translated_pos = helper_functions.random_translate_img(pos_image, xy_range_translate)
        translated_pos = prepare_image_for_net3D(translated_pos)
        img_list.append(translated_pos)
        node_predict.append(1)       
        out_class.append(int(info["diagnosis"]))
        
        rotated_pos = helper_functions.random_rotate_img(pos_image,0.6,30,230)
        rotated_pos = prepare_image_for_net3D(rotated_pos)
        img_list.append(rotated_pos)
        node_predict.append(1)       
        out_class.append(int(info["diagnosis"]))
        
        flipped_pos = helper_functions.random_flip_img(pos_image,0.5,0.5)
        flipped_pos = prepare_image_for_net3D(flipped_pos)
        img_list.append(flipped_pos)
        node_predict.append(1)       
        out_class.append(int(info["diagnosis"]))
        
        scaled_neg = helper_functions.random_scale_img(neg_image, xy_range_scale, lock_xy=False)
        scaled_neg = prepare_image_for_net3D(scaled_neg)
        img_list.append(scaled_neg)
        node_predict.append(0)       
        out_class.append(0)
               
        rotated_neg = helper_functions.random_rotate_img(neg_image,0.6,30,230)
        rotated_neg = prepare_image_for_net3D(rotated_neg)
        img_list.append(rotated_neg)
        node_predict.append(0)       
        out_class.append(0)
        
        flipped_neg = helper_functions.random_flip_img(neg_image,0.5,0.5)
        flipped_neg = prepare_image_for_net3D(flipped_neg)
        img_list.append(flipped_neg)
        node_predict.append(0)       
        out_class.append(0)
        
    random.shuffle(img_list)
    random.shuffle(node_predict)
    random.shuffle(out_class)
    
    while True:
        batch_indexes = np.random.choice(len(img_list),size=batch_size)
        batch_img_list = [img_list[i] for i in batch_indexes]
        batch_node_predict = [node_predict[i] for i in batch_indexes]
        batch_out_class = [out_class[i] for i in batch_indexes]
        x = np.vstack(batch_img_list)
        y_node_predict = np.vstack(batch_node_predict)
        y_class = np.vstack(batch_out_class)
        yield x, np.concatenate([y_node_predict, y_class],axis=-1)
    

def neural_network(input_shape=(CUBE_SIZE,CUBE_SIZE,CUBE_SIZE,1), load_weight_path = None):
    
    loss_function = CustomLoss()
    metrics = [loss_function.l_obj, loss_function.l_class] 
    inputs = Input(shape=input_shape, name="input_1")
    x = inputs
    
    # 1st layer group
    x = AveragePooling3D(pool_size=(2,1,1) , padding="same", strides=(2,1,1))(x)
    x = Conv3D(64, (3, 3, 3), padding="same", activation="relu", name="conv1")(x)
    x = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding="valid", name="pool1")(x)
    
    # 2nd layer group
    x = Conv3D(128, (3, 3, 3), padding="same", activation="relu", name="conv2")(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding="valid", name="pool2")(x)
    x = Dropout(rate=0.5)(x)
    
    # 3rd layer group
    x = Conv3D(256, (3, 3, 3), padding="same", activation="relu", name="conv3a")(x)
    x = Conv3D(256, (3, 3, 3), padding="same", activation="relu", name="conv3b")(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding="valid", name="pool3")(x)
    x = Dropout(rate=0.5)(x)
    
    # 4th layer group
    x = Conv3D(512, (3, 3, 3), padding="same", activation="relu", name="conv4a")(x)
    x = Conv3D(512, (3, 3, 3), padding="same", activation="relu", name="conv4b")(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding="valid", name="pool4")(x)
    x = Dropout(rate=0.5)(x)
    
    lastconv = Conv3D(64, (2, 2, 2), activation="relu", name="lastconv")(x)
    out_class = Conv3D(1, (1, 1, 1), activation="sigmoid", name="out_class1")(lastconv)
    out_class = Flatten(name="out_class2")(out_class)
    
    node_predict = Conv3D(1, (1, 1, 1), activation="sigmoid", name="node_predict1")(lastconv)
    node_predict= Flatten(name="node_predict2")(node_predict)
    
    model_out = concatenate([node_predict, out_class])
    model = Model(inputs=inputs, outputs=model_out)
    model.summary()
    
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
    model.compile(optimizer=Adam(lr=0.0001), loss=loss_function, metrics=metrics)
    return model

def train_on_data():
    batch_size = 10
    train_gen = data_generator(os.path.join(setting.MAIN_DIRECTORY,'Train'), batch_size)
    model = neural_network(load_weight_path = os.path.join(setting.MODEL_WEIGHTS,"model_luna16_full__fs_best.hd5"))
    model.fit_generator(train_gen,steps_per_epoch=420/batch_size,epochs=200)
    model.save(os.path.join(setting.MODEL_WEIGHTS,"my_model_1_end.hd5"))
