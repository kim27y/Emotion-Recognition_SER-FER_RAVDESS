import re
import os
import pandas as pd
import cv2
import random
import argparse
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Video preprocess arguments')

parser.add_argument('--path',
                    type=str,
                    default='./',
                    help='data path')

parser.add_argument('--task',
                    type=str,
                    default='all',
                    help='onlyface, onlyblack, all')

emotions_tras = {1:1, 2:4, 3:5, 4:0, 5:3, 6:2, 7:6}
emotions = {0:'angry', 1:'calm', 2:'disgust', 3:'fear', 4:'happy', 5:'sad', 6:'surprise'}

val_actors = ['19', '20']
test_actors = ['21', '22', '23', '24']

# data 디렉토리의 모든 video filename, feat, label, path 추출 및 반환 함수
def make_filename(path):
    filenames = []
    feats = []
    labels = []
    paths = []

    for (dirpath, dirnames, fn) in os.walk(path):
        for name in fn:
            filename = name.split('.')[0]
            feat = filename.split('-')[2:]
            label = feat[0]
            filenames.append(filename)
            feats.append(feat)
            labels.append(label)
            paths.append(os.path.join(dirpath,filename))
            
    return filenames, feats, labels, paths

# 비디오 데이터 프레이밍 후 저장, 경로 : './Datasets,RAVDESS_frames
def framing(filenames, paths, skip=1):
    for count, video in tqdm(enumerate(zip(filenames, paths)), desc='framing progress', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'):
        # Gather all its frames
        filename, input_path, output_path = video[0], video[1], video[1].replace('RAVDESS', 'RAVDESS_frames')
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        count = 0
        while os.path.exist(os.path.join(output_path, f'{filename}_{count}.png')):
            count += skip
            
        cap = cv2.VideoCapture(input_path + '.mp4')
        
        try:
        # Loop through all frames
            while True:
                # Capture frame
                ret, frame = cap.read()
                if (count % skip == 0 and count > 20):
                    #print(frame.shape)
                    if not ret:
                        break
                    frame = cv2.resize(frame, (398, 224))
                    cv2.imwrite(os.path.join(output_path, f'{filename}_{count}.png'), frame)
                count += 1
        finally:
            cap.release()

# 비디오 얼굴 framing 및 background blacking 진행 함수
def blacking(filenames, paths, skip=1):
    for count, video in tqdm(enumerate(zip(filenames, paths)), desc='blacking progress', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'):
        # Gather all its frames
        filename, input_path, output_path = video[0], video[1], video[1].replace('RAVDESS', 'RAVDESS_frames_black')
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        count = 0

        while os.path.exist(os.path.join(output_path, f'{filename}_{count}.png')):
            count += skip
            
        cap = cv2.VideoCapture(input_path + '.mp4')
        
        try:
        # Loop through all frames
            while True:
                # Capture frame
                ret, frame = cap.read()
                if (count % skip == 0 and count > 20):
                    #print(frame.shape)
                    if not ret:
                        break
                    #####
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                  # background from white to black
                    ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
                    frame[thresh == 255] = 0
                    #####
                    frame = cv2.resize(frame, (398, 224))
                    frame = frame[0:224, 87:311]
                    cv2.imwrite(os.path.join(output_path, f'{filename}_{count}.png'), frame)
                count += 1
        finally:
            cap.release()

# blacking + detection 함수
def onlyfaceBW(filenames, paths, skip=1):
    
    for count, video in tqdm(enumerate(zip(filenames, paths)), desc='onlyfaceBW progress', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'):
        # Gather all its frames
        filename, input_path, output_path = video[0], video[1], video[1].replace('RAVDESS', 'RAVDESS_frames_face_BW')
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        haar_cascade = cv2.CascadeClassifier('./Other/haarcascade_frontalface_default.xml')
        count = 0
        
        while os.path.exist(os.path.join(output_path, f'{filename}_{count}.png')):
            count += skip
        
        cap = cv2.VideoCapture(input_path + '.mp4')
        
        try:
        # Loop through all frames
            while True:
                # Capture frame
                ret, frame = cap.read()
                if os.path.exists(os.path.join(output_path, f'{filename}_{count}.png')):
                    count += 1
                    continue
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
                            print(f"error {filename} {count} 발생!, 얼굴 인식 불가!")
                            break
                    if len(faces) > 1:
                        ex = []
                        print(type(faces))
                        for elem in faces:
                            for (x, y, w, h) in [elem]:
                                ex.append(frame[y:y + h, x:x + w])


                    for (x, y, w, h) in faces:
                        face = frame[y:y + h, x:x + w]

                    face = cv2.resize(face, (234, 234))
                    face = face[5:-5, 5:-5]
                    cv2.imwrite(os.path.join(output_path, f'{filename}_{count}.png'), face)
                    
                count += 1
        finally:
            cap.release()
        
def sampling(list, num_frames_desired):
    tot = []
    for elem in list:
        sampled_list = random.sample(elem, num_frames_desired)
        tot += sampled_list
    return(tot)

def preprocess(path, task, random_seed):
    filenames, feats, labels, paths = make_filename(path)
    frames_per_vid = 20
    filenames_train = [] # train
    faces = []
    
    if task == 'onlyface':
        framing(filenames, paths, 3)
        dataset_path = os.path.join('Datasets','RAVDESS_frames')
        
    elif task == 'onlyblack':
        blacking(filenames, paths, 3)
        dataset_path = os.path.join('Datasets','RAVDESS_frames_black')

    elif task == 'all':
        onlyfaceBW(filenames, paths, 3)
        dataset_path = os.path.join('Datasets','RAVDESS_frames_face_BW')

    else:
        raise ValueError("invalid task")
    
    for (dirpath, dirnames, fn) in os.walk(dataset_path):
        if fn != []:
            class_temp = int(fn[0].split('-')[2]) - 1
            if class_temp != 0:                                                     # exclude 'neutral' label
                if any(act in dirpath for act in (test_actors+val_actors))==False:  # select only train actors
                    path = [os.path.join(dirpath, elem) for elem in fn]
                    label = [emotions_tras[class_temp]] * len(fn)                   # emotion transposition
                    filenames_train.append(list(zip(path, label)))
    
    filenames_sampled = sampling(filenames, frames_per_vid)
    random.shuffle(filenames_sampled)

    for path, label in tqdm(filenames_sampled):
        face = cv2.imread(path)
        face = cv2.resize(face, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        faces.append(face)
    
    faces = np.array(faces)
    mean_face = np.mean(faces, axis=0)
    mean_face = mean_face/255
    mean_face = np.expand_dims(mean_face, axis=2)
    np.save(os.path.join('Other','mean_face.npy'), mean_face)

if __name__ == "__main__":
    args = parser.parse_args()
    path = args.path
    task = args.task
    
    preprocess(path, task, 42)