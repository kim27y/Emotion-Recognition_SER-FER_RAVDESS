# Utility
from google.colab import drive
from shutil import copyfile, copy
import matplotlib.pyplot as plt
import IPython.display as ipd
import os
import pandas as pd
import numpy as np
import itertools

# Audio processing
import librosa
import librosa.display
import audiomentations

# Sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Keras
import keras
from keras import regularizers
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Dense, Activation, LeakyReLU, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

parser = argparse.ArgumentParser(description='Start Training')
parser.add_argument('--data_path',
                    type=str,
                    default='./Datasets',
                    help='data sort, train or test or all')
parser.add_argument('--batch_size',
                    type=str,
                    default='64',
                    help='data sort, train or test or all')
parser.add_argument('--epoch',
                    type=str,
                    default='10',
                    help='data sort, train or test or all')
parser.add_argument('--lr',
                    type=str,
                    default='0.001',
                    help='data sort, train or test or all')
parser.add_argument('--model',
                    type=str,
                    default='model5',
                    help='data sort, train or test or all')





if tf.test.gpu_device_name(): 
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
    

def switch(emotion):
  if emotion == 2:
    return 'calm'
  elif emotion == 3:
    return 'happy'
  elif emotion == 4:
    return 'sad'
  elif emotion == 5:
    return 'angry'
  elif emotion == 6:
    return 'fear'
  elif emotion == 7:
    return 'disgust'
  elif emotion == 8:
    return 'surprise'


def make_dataset(paths):
    emotion = []
    path = []
    
    for i in paths:
        filename = os.listdir('./Datasets/AUDIO/' + i)
        for f in filename:
            # Remove wav extension
            id = f[:-4].split('-')
            if(id[2] != '01'):
            # Dividing according to emotions
                emotion.append(switch(int(id[2])))
                path.append('./Datasets/AUDIO/' + i + '/' + f)
    
    df = pd.concat([pd.DataFrame(emotion), pd.DataFrame(path)], axis = 1)
    df.columns = ['emotion', 'path']
    
    return df

def feature_extractor(input, feature, sr = 48000):
  # Mel Spectrogram
  if(feature == 'mel'):
    return librosa.power_to_db(librosa.feature.melspectrogram(input*1.0, sr = sr, n_fft = 1024, n_mels = 128, fmin = 50, fmax = 24000)) 

def process_data(paths):
    df = make_dataset(paths)
    
    audio = []
    for filename in df['path']:
        data, sampling_rate = librosa.load(filename, sr = 48000, duration = 3, offset = 0.5) # We want the native sr
        audio.append(data)
    df = pd.DataFrame(np.column_stack([df, audio]))
    df.columns = ['emotion', 'path', 'data']
    
    for i in range(len(df)):
        if(len(df['data'][i]) != 144000):
            start_pad = (144000 - len(df['data'][i]))//2
            end_pad = 144000 - len(df['data'][i]) - start_pad
            df['data'][i] = np.pad(df['data'][i], (start_pad, end_pad), mode = 'constant')
            
    df['features'] = [0] * 1344
    for i in range(len(df)):
        mel = feature_extractor(df['data'][i], 'mel')
        df['features'][i] = np.array(mel, dtype = object)        
        
    X_test = df['features'][1120:1344].tolist()
    y_test = df['emotion'][1120:1344].tolist()
    
    X_train = df['features'][:1120].tolist()
    y_train = df['emotion'][:1120].tolist()
    
    noise = audiomentations.Compose([
        audiomentations.AddGaussianNoise(p = 1)
    ])
    
    pitchShift = audiomentations.Compose([
        audiomentations.PitchShift(p = 1)
    ])
    
    stretch = audiomentations.Compose([
        audiomentations.TimeStretch(p = 1)
    ])
    
    shift = audiomentations.Compose([
        audiomentations.Shift(min_fraction = 0.25, max_fraction = 0.25, rollover = False, p = 1)
    ])
    
    for i in range(1120):
        augmented_samples_1 = noise(df['data'][i], 48000)
        augmented_samples_2 = pitchShift(df['data'][i], 48000)
        augmented_samples_3 = stretch(df['data'][i], 48000)
        augmented_samples_4 = shift(df['data'][i], 48000)
        
        mel_1 = feature_extractor(augmented_samples_1, 'mel')
        mel_2 = feature_extractor(augmented_samples_2, 'mel')
        mel_3 = feature_extractor(augmented_samples_3, 'mel')
        mel_4 = feature_extractor(augmented_samples_4, 'mel')
        
        X_train.append(np.array(mel_1, dtype = object))
        y_train.append(df['emotion'][i])
        
        X_train.append(np.array(mel_2, dtype = object))
        y_train.append(df['emotion'][i])
        
        X_train.append(np.array(mel_3, dtype = object))
        y_train.append(df['emotion'][i])
        
        X_train.append(np.array(mel_4, dtype = object))
        y_train.append(df['emotion'][i])
    
    for i in range(len(X_train)):
        X_train[i] = X_train[i].astype(np.float64)
    for i in range(len(X_test)):
        X_test[i] = X_test[i].astype(np.float64)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    scaler_train = StandardScaler()

    X_train[:1120] = scaler_train.fit_transform(X_train[:1120].reshape(-1, X_train.shape[-1])).reshape(X_train[:1120].shape)
    X_train[1120:] = scaler_train.transform(X_train[1120:].reshape(-1, X_train.shape[-1])).reshape(X_train[1120:].shape)
    X_test = scaler_train.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    joblib.dump(scaler_train, './Datasets/std_scaler.bin')
    
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))
    
    X_train = np.expand_dims(X_train, axis = 3)
    X_test = np.expand_dims(X_test, axis = 3)
    
    np.save('./Datasets/X_test', X_test)
    np.save('./Datasets/y_test', y_test)
    
    return X_train, X_test, y_train, y_test
    


def Models():
    num_classes = 7

    model31 = Sequential()

    model31.add(Conv2D(64, (3, 3), activation = 'relu', input_shape = (128, 282, 1)))
    model31.add(MaxPooling2D((2, 2)))

    model31.add(Conv2D(128, (3, 3), activation = 'relu', kernel_regularizer = l2(0.001), bias_regularizer = l2(0.01)))
    model31.add(MaxPooling2D((2, 2)))
    model31.add(Dropout(0.2))

    model31.add(Conv2D(128, (3, 3), activation = 'relu', kernel_regularizer = l2(0.001), bias_regularizer = l2(0.01)))
    model31.add(GlobalMaxPooling2D())
    model31.add(Dropout(0.2))

    model31.add(Dense(128, activation = 'relu', kernel_regularizer = l2(0.001), bias_regularizer = l2(0.01)))
    model31.add(Dropout(0.2))

    model31.add(Dense(num_classes, activation = 'softmax'))
    
    return model31

def train(paths):
    model31 = Model()
    
    X_train, X_test, y_train, y_test = process_data(paths)
    
    model31.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = 0.001), metrics = ['accuracy'])
    history31 = model31.fit(X_train, y_train, batch_size = 32, epochs = 40, verbose = 0)
    
    y_pred = model31.predict(X_test)
    y_pred_ = np.argmax(y_pred, axis = 1)
    y_test_ = np.argmax(y_test, axis = 1)
    print(classification_report(y_test_, y_pred_))
    
    cm = confusion_matrix(y_test_, y_pred_)
    plt.imshow(cm, cmap = plt.cm.Blues)
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    
    print(top_k_accuracy_score(y_test_, y_pred, k = 2))
    print(top_k_accuracy_score(y_test_, y_pred, k = 3))
    
    model31.save('./Models/Audio Stream/Audio_CNN 2D/model3_1.h5')


    
if __name__ == '__main__':
    args = parser.parse_args()
    path = args.data_path
    
    train(path)    
