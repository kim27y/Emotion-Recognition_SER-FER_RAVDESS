import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime

import tensorflow as tf

# Keras module and tools
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Input, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dense, Dropout, RandomFlip
from keras.metrics import sparse_categorical_accuracy, sparse_top_k_categorical_accuracy
from keras.losses import sparse_categorical_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras_cv.layers import RandomCutout
import keras_tuner

if tf.test.gpu_device_name(): 
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
    
emotions_tras = {1:1, 2:4, 3:5, 4:0, 5:3, 6:2, 7:6} # to match the audio stream labels
emotions = {0:'angry', 1:'calm', 2:'disgust', 3:'fear', 4:'happy', 5:'sad', 6:'surprise'}

path_frames_face_BW = os.path.join('Datasets','RAVDESS_frames_face_BW')

height_orig = 224
width_orig = 224
height_targ = 112
width_targ = 112

num_classes = len(emotions)

val_actors = ['19', '20']
test_actors = ['21', '22', '23', '24']

def make_dataset():
    filenames_train = [] # train
    filenames_val = [] # validation

    for (dirpath, dirnames, fn) in os.walk(path_frames_face_BW):
        if fn != []:
            class_temp = int(fn[0].split('-')[2]) - 1
            if class_temp != 0:                                                     # exclude 'neutral' label
                if any(act in dirpath for act in (test_actors+val_actors))==False:  # select only train actors
                    path = [os.path.join(dirpath, elem) for elem in fn]
                    label = [emotions_tras[class_temp]] * len(fn)                   # emotion transposition
                    filenames_train.append(list(zip(path, label)))
                
                if any(act in dirpath for act in val_actors):                       # select only validation actors
                    path = [os.path.join(dirpath, elem) for elem in fn]
                    label = [emotions_tras[class_temp]] * len(fn)
                    filenames_val.append(list(zip(path, label)))
    
    return filenames_train, filenames_val

def sampling(list, num_frames_desired):
    tot = []
    for elem in list:
        sampled_list = random.sample(elem, num_frames_desired)
        tot += sampled_list
    return(tot)

def parse_image_4(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float16)
    image = tf.image.resize_with_crop_or_pad(image, height_orig, width_orig)
    image = tf.image.resize(image, [height_targ, width_targ])
    print('shape frames:', image.shape)
    return image

def parse_image_5(filename):
    mean_face = np.load(os.path.join('Other','mean_face.npy'))
    mean_face = tf.convert_to_tensor(mean_face, dtype=tf.float32)
    
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float16)
    image = tf.image.resize_with_crop_or_pad(image, height_orig, width_orig)
    image = tf.image.resize(image, [height_targ, width_targ])
    image = image - mean_face
    print('shape frames:', image.shape)
    return image

def configure_for_performance(ds,batch_size):
    ds = ds.shuffle(buffer_size=1000) # serve?
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def load_dataset(filenames, batch_size,model):
    random.seed(42)
    
    frames_per_vid = min([len(elem) for elem in filenames])     # number of frames per clip in order to have balanced classes
    print("frames per video:", frames_per_vid) 

    filenames_sampled = sampling(filenames, frames_per_vid)
    random.shuffle(filenames_sampled)

    zipped = [list(t) for t in zip(*filenames_sampled)]

    names = zipped[0]
    labels = zipped[1]

    names = tf.data.Dataset.from_tensor_slices(names)
    if model=='model4':
        images = names.map(parse_image_4, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        images = names.map(parse_image_5, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    labels = [elem for elem in labels]
    labels = tf.data.Dataset.from_tensor_slices(labels)

    ds = tf.data.Dataset.zip((images, labels))
    ds = configure_for_performance(ds)

    frame_number = len(filenames_sampled)
    step_per_epoch = frame_number // batch_size
    print('frames number:', frame_number, '\nbatch size:', batch_size, '\nbatch number:', step_per_epoch)
    return ds, step_per_epoch

def Model(model):
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
    
    input = Input(shape=(width_targ, height_targ, 1))
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

def train(train_data, val_data,batch_size, epochs=10, lr=0.001, momentum=0.5,model='model4'):
    train_ds, step_per_epoch_train = load_dataset(train_data, batch_size)
    val_ds, step_per_epoch_val = load_dataset(val_data, batch_size)
    
    checkpoint_filepath = f'./Models/Video_stream/video_model_{datetime.now().strftime("%d-%m-%y_%H-%M")}_' + '[{val_sparse_categorical_accuracy:.4f}]_face.hdf5'

    reduce_lr = ReduceLROnPlateau(monitor="val_sparse_categorical_accuracy", factor=0.5, patience=2, verbose=1)
    early_stop = EarlyStopping( monitor="val_sparse_categorical_accuracy", patience=4, verbose=1, restore_best_weights=True)
    save_best = ModelCheckpoint(filepath=checkpoint_filepath,
                                                monitor='val_sparse_categorical_accuracy',
                                                mode='max',
                                                save_best_only=True)
    

    
    net = Model(model)
    
    net.compile(
        optimizer = Adam(learning_rate=lr),
        # optimizer=keras.optimizers.SGD(learning_rate=learningrate, momentum=momentum)
        loss = sparse_categorical_crossentropy,
        metrics = [sparse_categorical_accuracy],
    )

    history = net.fit(train_ds,
                    epochs=epochs,
                    validation_data=val_ds,
                    batch_size=batch_size,
                    steps_per_epoch=step_per_epoch_train,
                    validation_steps=step_per_epoch_val,
                    # callbacks=[reduce_lr, early_stop, save_best],
                    # callbacks=[save_best],
                    verbose=1)
        
    net.evaluate(val_ds,
            batch_size=batch_size,
            steps=step_per_epoch_val)
    
    return history

    
    
    