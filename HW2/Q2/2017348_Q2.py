#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:55:33 2020

@author: kritverma
"""


import numpy as np
from scipy.io import wavfile
import scipy.fftpack as fft
import glob
import scipy
import pickle
import pandas as pd
from scipy.stats import skew 
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import math
from random import randint as randi

sample_rate, audio = wavfile.read("test.wav")
noise_data=pickle.load(open('../noise_data', 'rb'))

# -----------------------------------------------------VARIABLES-----------------------------------------
FFT_size=2048
hop_size=15

high=2595.0
low=700.0

freq_min = 0
freq_high = sample_rate / 2
mel_filter_num = 10

param_grid = {'C': [0.0001,0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0000001], 
              'kernel': ['rbf']}  

# -----------------------------------------------------HELPER FUNCTIONS-----------------------------------------

def nan_check(val):
    if (math.isnan(val)):
        return 0
    elif(val==float('inf') or val==float('-inf')):
        return 0
    else:
        return val
    
def create_feature_vector(mfcc):
    all_feature_vectors=[]
    for i in range(mfcc.shape[0]):
        feature_vector=[]
        feature_vector.append(nan_check(np.max(mfcc[i])))
        feature_vector.append(nan_check(np.min(mfcc[i])))
        feature_vector.append(nan_check(np.mean(mfcc[i])))
        feature_vector.append(nan_check(np.median(mfcc[i])))
        feature_vector.append(nan_check(np.std(mfcc[i])))
        feature_vector.append(nan_check(skew(mfcc[i])))
        all_feature_vectors.append(feature_vector)
    return all_feature_vectors

def create_noise_audio(audio):
    noise=noise_data[randi(0,5)]
    if(len(audio)>len(noise)):
        noise_audio=audio[:len(noise)]+0.1*noise
        return noise_audio
    else:
        noise_audio=audio+0.1*noise[:len(audio)]
        return noise_audio


# -----------------------------------------------------MFCC FUNCTIONS-----------------------------------------

def frame_audio(audio,sample_rate):
    pad_size=FFT_size // 2
    audio = np.pad(audio, pad_size, mode='reflect')
    frame_len=(sample_rate * hop_size) // 1000
    frame_len = np.round(frame_len)
    
    frame_num = (len(audio) - FFT_size) // frame_len
    frame_num+=1
    
    frames = np.zeros((frame_num,FFT_size))
    
    for n in range(frame_num):
        start=n*frame_len
        end=start+FFT_size
        frames[n] = audio[start:end]
    
    return frames

def freq_to_mel(freq):
    freq= freq/low
    freq+=1
    log_freq=np.log10(freq)
    return  high * log_freq

def met_to_freq(mels):
    mels=mels/high
    mels=np.power(10.0,mels) - 1.0
    return low * (mels)

def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = met_to_freq(mels)
    
    filter_points= np.floor((FFT_size + 1) / sample_rate * freqs).astype(int)
    
    return filter_points, freqs

def get_filters(filter_points, FFT_size):
    start=len(filter_points)
    stop=FFT_size//2
    filters = np.zeros((start-2,stop+1))
    
    for n in range(start-2):
        fil_n=filter_points[n]
        fil_n1=filter_points[n + 1]
        fil_n2= filter_points[n + 2]
        filters[n, fil_n : fil_n1] = np.linspace(0, 1, fil_n1 - fil_n)
        filters[n, fil_n1 : fil_n2] = np.linspace(1, 0, fil_n2 - fil_n1)
        
    return filters

def dct(filter_len):
    
    dct_filter_num = 12
    dim=(dct_filter_num,filter_len)
    arr = np.empty((dim[0],dim[1]))
    arr[0, :] = 1.0 / np.sqrt(dim[1])
    
    start=1
    stop=2 * dim[1]
    step=2
    multiplier=np.pi / (2.0 * dim[1])
    samples = np.arange(start, stop , step) * multiplier

    for i in range(1, dim[0]):
        arr[i, :] = np.cos(i * samples) * np.sqrt(2.0 / dim[1])
        
    return arr

def filter_fft(audio_fft,T_audio_framed):
    
    for n in range(audio_fft.shape[1]):
        audio_fft[:, n] = fft.fft(T_audio_framed[:, n], axis=0)[:audio_fft.shape[0]]
    return audio_fft

def get_filtered_audio(filters,audio_power):
    audio_filtered = np.dot(filters, np.transpose(audio_power))
    audio_log = 10.0 * np.log10(audio_filtered)
    return audio_log

def create_mfcc(audio):
    audio = audio / np.max(np.abs(audio))
    
    audio_framed = frame_audio(audio, sample_rate)
    
    T_audio_framed = np.transpose(audio_framed)
    
    row_size=FFT_size // 2
    col_size= T_audio_framed.shape[1]
    
    audio_fft = np.empty((row_size+1,col_size), dtype=np.complex64, order='F')

    audio_fft=filter_fft(audio_fft,T_audio_framed)
    
    audio_fft = np.transpose(audio_fft)

    audio_power = np.square(np.abs(audio_fft))
    
    filter_points_set = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate)
    filter_points, mel_freqs= filter_points_set[0], filter_points_set[1]
    
    filters = get_filters(filter_points, FFT_size)

    audio_log=get_filtered_audio(filters,audio_power)    
    
    dct_filters = dct(mel_filter_num)

    mfcc_coefficents = np.dot(dct_filters, audio_log)
    
    return mfcc_coefficents


def create_noise_training_model():
    count=0
    mfcc_db={}
    train_query=glob.glob('../training/*')
    for i in train_query:
        query_folder_path=i+"/*.wav"
        query_folder=glob.glob(query_folder_path)
        for j in query_folder:
            count+=1
            print(count)
            audio_name=j.split("/",2)[2].replace(".wav","")
            sample_rate, audio = wavfile.read(j)
            noise_audio=create_noise_audio(audio)
            mfcc = create_mfcc(noise_audio)
            feature_vec=create_feature_vector(mfcc)
            audio_spec={}
            audio_spec['sample_rate']= sample_rate
            audio_spec['mfcc']= mfcc
            audio_spec['feature_vectors']= feature_vec
            mfcc_db[audio_name]=audio_spec  
    return mfcc_db

def create_training_model():
    count=0
    mfcc_db={}
    train_query=glob.glob('../training/*')
    for i in train_query:
        query_folder_path=i+"/*.wav"
        query_folder=glob.glob(query_folder_path)
        for j in query_folder:
            count+=1
            print(count)
            audio_name=j.split("/",2)[2].replace(".wav","")
            sample_rate, audio = wavfile.read(j)
            mfcc = create_mfcc(audio)
            feature_vec=create_feature_vector(mfcc)
            audio_spec={}
            audio_spec['sample_rate']= sample_rate
            audio_spec['mfcc']= mfcc
            audio_spec['feature_vectors']= feature_vec
            mfcc_db[audio_name]=audio_spec  
    return mfcc_db

def create_validation_model():
    count=0
    validation_mfcc={}
    train_query=glob.glob('../validation/*')
    for i in train_query:
        query_folder_path=i+"/*.wav"
        query_folder=glob.glob(query_folder_path)
        for j in query_folder:
            count+=1
            print(count)
            audio_name=j.split("/",2)[2].replace(".wav","")
            sample_rate, audio = wavfile.read(j)
            mfcc = create_mfcc(audio)
            feature_vec=create_feature_vector(mfcc)
            audio_spec={}
            audio_spec['sample_rate']= sample_rate
            audio_spec['mfcc']= mfcc
            audio_spec['feature_vectors']= feature_vec
            validation_mfcc[audio_name]=audio_spec  
    return validation_mfcc


def check_noise_training_model():
    try:
        mfcc_db=pickle.load(open('feature_vectors_mfcc_noise', 'rb'))
        print("Fetched Noise Trained Model")
        return mfcc_db
    except:
        print("Training Noise Model")
        mfcc_db=create_noise_training_model()
        pickle.dump(mfcc_db,open('feature_vectors_mfcc_noise', 'wb'))
        return mfcc_db

def check_training_model():
    try:
        mfcc_db=pickle.load(open('feature_vectors_mfcc', 'rb'))
        print("Fetched Trained Model")
        return mfcc_db
    except:
        print("Training Model")
        mfcc_db=create_training_model()
        pickle.dump(mfcc_db,open('feature_vectors_mfcc', 'wb'))
        return mfcc_db

def check_validation_model():
    try:
        validation_mfcc=pickle.load(open('validation_feature_mfcc', 'rb'))
        print("Fetched Validation Model")
        return validation_mfcc
    except:
        print("Training Validation Model")
        validation_mfcc=create_validation_model()
        pickle.dump(validation_mfcc,open('validation_feature_mfcc', 'wb'))
        return validation_mfcc
    
# sample_rate, audio = wavfile.read("test.wav")
# noise_audio=create_noise_audio(audio)
# mfcc=create_mfcc(audio)
# feature_set=create_feature_vector(mfcc)


mfcc_db=check_training_model()
mfcc_noise_db=check_noise_training_model()
validation_mfcc=check_validation_model()


training_df=pd.DataFrame.from_dict(mfcc_db,orient='index')
training_noise_df=pd.DataFrame.from_dict(mfcc_noise_db,orient='index')
validation_df=pd.DataFrame.from_dict(validation_mfcc,orient='index')

def process_data(dataframe):
    data=pd.DataFrame(columns=['class_name', 'feature_vectors'])
    data_index=dataframe.index

    for i in range(dataframe.shape[0]):
        data.loc[i]= [data_index[i].split("/")[0],np.asarray(dataframe.iloc[i]['feature_vectors'])]
    
    return data

features_train=process_data(training_df)
features_noise_train=process_data(training_noise_df)
features_val=process_data(validation_df)

label_encoder = preprocessing.LabelEncoder()
features_train['class_name']= label_encoder.fit_transform(features_train['class_name'])
features_noise_train['class_name']= label_encoder.fit_transform(features_noise_train['class_name'])
features_val['class_name']= label_encoder.fit_transform(features_val['class_name'])


x_train=[]
y_train=[]
x_noise_train=[]
y_noise_train=[]
x_val=[]
y_val=[]
for i in range(features_train.shape[0]):
    x_train.append(features_train.iloc[i]['feature_vectors'].flatten())

for i in range(features_train.shape[0]):
    y_train.append(features_train.iloc[i]['class_name'].flatten())
    
for i in range(features_noise_train.shape[0]):
    x_noise_train.append(features_noise_train.iloc[i]['feature_vectors'].flatten())

for i in range(features_noise_train.shape[0]):
    y_noise_train.append(features_noise_train.iloc[i]['class_name'].flatten())

for i in range(features_val.shape[0]):
    x_val.append(features_val.iloc[i]['feature_vectors'].flatten())

for i in range(features_val.shape[0]):
    y_val.append(features_val.iloc[i]['class_name'].flatten())

def get_saved_grid():
    try:
        grid=pickle.load(open('mfcc_grid', 'rb'))
        print("Fetched Saved Grid")
        return grid
    except:
        print("Training Grid ")
        grid = GridSearchCV(SVC(), param_grid, verbose = 3) 
        grid.fit(x_train, y_train)
        pickle.dump(grid,open('mfcc_grid', 'wb'))
        return grid
    
def get_saved_noise_grid():
    try:
        noise_grid=pickle.load(open('mfcc_noise_grid', 'rb'))
        print("Fetched Noise Saved Grid")
        return noise_grid
    except:
        print("Training Noise Grid ")
        noise_grid = GridSearchCV(SVC(), param_grid, verbose = 3) 
        noise_grid.fit(x_noise_train, y_noise_train)
        pickle.dump(noise_grid,open('mfcc_noise_grid', 'wb'))
        return noise_grid
    
grid=get_saved_grid()
noise_grid=get_saved_noise_grid()

noise_grid_predictions = noise_grid.predict(x_val)
grid_predictions = grid.predict(x_val)
print(classification_report(y_val, grid_predictions,zero_division=0))
print(classification_report(y_val, noise_grid_predictions,zero_division=0))