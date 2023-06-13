import numpy as np
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime
import argparse
import tensorflow as tf
import re
import pandas as pd
import cv2
import random
from sklearn.decomposition import PCA
from tqdm import tqdm
from tensorflow.keras.models import load_model

# Keras module and tools
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Input, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dense, Dropout, RandomFlip
from keras.metrics import sparse_categorical_accuracy, sparse_top_k_categorical_accuracy
from keras.losses import sparse_categorical_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras_cv.layers import RandomCutout
import keras_tuner

parser = argparse.ArgumentParser(description='setup database')
parser.add_argument('--inference_path',
                    type=str,
                    default='./Datasets/RAVDESS/datas/Actor_01/01-02-01-01-01-01-01.mp4',
                    help='data sort, train or test or all')
parser.add_argument('--checkpoint',
                    type=str,
                    default='./Models/Video_stream/video_model_13-06-23_11-56_model4_[0.5242]_face.hdf5',
                    help='data sort, train or test or all')

emotions_tras = {1:1, 2:4, 3:5, 4:0, 5:3, 6:2, 7:6} # to match the audio stream labels
emotions = {0:'angry', 1:'calm', 2:'disgust', 3:'fear', 4:'happy', 5:'sad', 6:'surprise'}

num_classes = len(emotions)

def Models(model):
    if model == 'model4':
        dropout = 0.5
        
        data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomCutout(0.4, 0.4, fill_mode="constant", fill_value=0.0, seed=None)
        ])
    
    else:
        dropout = 0.4
        data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal")
        ])
    
    input = Input(shape=(112, 112, 1))
    x = input
    x = data_augmentation(x)

    x = Conv2D(filters=32, kernel_size=(5,5), padding='same', kernel_initializer='he_normal', name='conv2d_0')(x)
    x = BatchNormalization(name='batchnorm_0')(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(2,2), name='maxpool2d_0')(x)

    x = Conv2D(filters=64, kernel_size=(5,5), padding='same', kernel_initializer='he_normal', name='conv2d_1')(x)
    x = BatchNormalization(name='batchnorm_1')(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(2,2), name='maxpool2d_1')(x)

    x = Dropout(dropout, name='dropout_1')(x)

    x = Conv2D(filters=128, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', name='conv2d_2')(x)
    x = BatchNormalization(name='batchnorm_2')(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(2,2), name='maxpool2d_2')(x)

    x = Dropout(dropout, name='dropout_2')(x)

    x = Conv2D(filters=256, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', name='conv2d_3')(x)
    x = BatchNormalization(name='batchnorm_3')(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(2,2), name='maxpool2d_3')(x)

    x = Dropout(0.5, name='dropout_3')(x)

    # x = Conv2D(filters=256, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', name='conv2d_4')(x)
    # x = BatchNormalization(name='batchnorm_x')(x)
    # x = Activation('elu')(x)
    # x = MaxPooling2D(pool_size=(2,2), name='maxpool2d_4')(x)

    # x = Dropout(0.5, name='dropout_4')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(128, kernel_initializer='he_normal', name='dense_1')(x)
    x = BatchNormalization(name='batchnorm_4')(x)
    x = Activation('elu')(x)

    x = Dropout(0.6, name='dropout_4')(x)

    x = Dense(num_classes, activation='softmax', name='out_layer')(x)

    output = x

    net_4 = Model(inputs=input, outputs=output)
    
    return net_4

def preprocessing(video):
    cap = cv2.VideoCapture(video)
    haar_cascade = cv2.CascadeClassifier('./Other/haarcascade_frontalface_default.xml')
    frames = []
    count = 0
    skip = 3
    
    try:
        # Loop through all frames
        while True:
            # Capture frame
            ret, frame = cap.read()
            if (count % skip == 0 and count > 20):
                #print(frame.shape)
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = haar_cascade.detectMultiScale(frame, scaleFactor=1.12, minNeighbors=9)
                # if len(faces) != 1:
                    
                if len(faces) == 0:
                    faces = haar_cascade.detectMultiScale(frame, scaleFactor=1.02, minNeighbors=9)
                    if len(faces) == 0:
                        continue
                if len(faces) > 1:
                    ex = []
                    for elem in faces:
                        for (x, y, w, h) in [elem]:
                            ex.append(frame[y:y + h, x:x + w])

                for (x, y, w, h) in faces:
                    face = frame[y:y + h, x:x + w]

                face = cv2.resize(face, (234, 234))
                face = face[5:-5, 5:-5]
                face = cv2.resize(face, (112, 112))
                face = np.expand_dims(face, axis=2)
                face = tf.cast(face, dtype=tf.float32)
                frames.append(face)
            count += 1
    finally:
        cap.release()

    frames = [i/255 for i in frames]
    frames = tf.data.Dataset.from_tensor_slices(frames)
    return frames

if __name__=='__main__':
    args = parser.parse_args()
    check_point= args.checkpoint
    path = args.inference_path
    model = load_model(path)
    # if check_point.split('_')[3] == 'model4':
    #     model = Model('model4')
    # else:
    #     model = Model('model5')
        
    # checkpoint = tf.train.Checkpoint(model=model)
    # # checkpoint.restore(check_point).expect_partial()
    
    data = preprocessing(path)
    for i in data:
        output = model.predict(i)
    
    ###########
    #결과#
    ###########