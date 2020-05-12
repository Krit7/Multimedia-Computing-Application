#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:42:20 2020

@author: kritverma


REFERENCE :- https://fairyonice.github.io/implement-the-spectrogram-from-scratch-in-python.html
"""

import warnings
from scipy.io import wavfile
import numpy as np
import glob
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
import random
import matplotlib.pyplot as plt
from random import randint as randi

warnings.filterwarnings("ignore", category=RuntimeWarning) 

L = 256
noise_data=pickle.load(open('../noise_data', 'rb'))

param_grid = {'C': [0.0001,0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0000001], 
              'kernel': ['rbf']}  
  
def create_noise_audio(audio):
    noise=noise_data[randi(0,5)]
    if(len(audio)>len(noise)):
        noise_audio=audio[:len(noise)]+0.1*noise
        return noise_audio
    else:
        noise_audio=audio+0.1*noise[:len(audio)]
        return noise_audio


def get_fourier_coeff(data):
    
    """
    
    k = kth freq
    N = total no sample
    n = time
    k//N = F  
    
    
    """
    mag = []
    N = len(data)
    nyq_lim=N//2
    k = np.arange(0,N)
    
    for n in range(nyq_lim): 
        e_x=np.exp((1j*2*np.pi*k*n)/N)
        x_k=np.sum(data*e_x)/N
        nyq_x_k=np.abs(x_k)*2
        mag.append(nyq_x_k)
    return mag


def get_Hz_scale_vec(ks,sample_rate,Npoindata):
    freq_Hz = ks*sample_rate/Npoindata
    freq_Hz  = freq_Hz.astype("int")
    return freq_Hz 

def rescale(xns):
    specX = np.array(xns).T
    return 10*np.log10(specX)

def create_spectrogram(data,NFFT):
    
    noverlap = NFFT//2
    stop=len(data)
    step=NFFT-noverlap
    stardata_range  = np.arange(0,stop,step,dtype=int)
    stardata=[]
    for i in stardata_range:
        if(i+NFFT<stop):
            stardata.append(i)
    coeff = []
    for i in stardata:
        coeff_lim=data[i : i + NFFT]
        data_window = get_fourier_coeff(coeff_lim) 
        coeff.append(data_window)
    spec = rescale(coeff)
    return(stardata,spec)


def plot_spectrogram(spec,sample_rate, L, stardata, data):
    mappable=None
    total_data_sec = len(data)/sample_rate
    plt_spec = plt.imshow(spec,origin='lower')
    
    plt.xlabel("Time (sec)")
    plt.ylabel("Frequency (Hz)")
    
    plt.title("Spectrogram")
    plt.colorbar(mappable,use_gridspec=True)
    plt.show()
    return(plt_spec)


def create_training_model():
    count=0
    spectogram_db={}
    train_query=glob.glob('../training/*')
    for i in train_query:
        query_folder_path=i+"/*.wav"
        query_folder=glob.glob(query_folder_path)
        for j in query_folder:
            count+=1
            print(count)
            audio_name=j.split("/",2)[2].replace(".wav","")
            sample_rate, data = wavfile.read(j)
            stardata, spec = create_spectrogram(data,L)
            audio_spec={}
            audio_spec['sample_rate']= sample_rate
            audio_spec['spectrogram']= spec
            audio_spec['stardata']= stardata
            audio_spec['audio_data']= data
            spectogram_db[audio_name]=audio_spec  
    return spectogram_db

def create_noise_training_model():
    count=0
    spectogram_db={}
    train_query=glob.glob('../training/*')
    for i in train_query:
        query_folder_path=i+"/*.wav"
        query_folder=glob.glob(query_folder_path)
        for j in query_folder:
            count+=1
            print(count)
            audio_name=j.split("/",2)[2].replace(".wav","")
            sample_rate, data = wavfile.read(j)
            noise_audio=create_noise_audio(data)
            stardata, spec = create_spectrogram(noise_audio,L)
            audio_spec={}
            audio_spec['sample_rate']= sample_rate
            audio_spec['spectrogram']= spec
            audio_spec['stardata']= stardata
            audio_spec['audio_data']= data
            spectogram_db[audio_name]=audio_spec  
    return spectogram_db

def create_validation_model():
    validation_spectogram={}
    train_query=glob.glob('../validation/*')
    for i in train_query:
        query_folder_path=i+"/*.wav"
        query_folder=glob.glob(query_folder_path)
        for j in query_folder:
            audio_name=j.split("/",2)[2].replace(".wav","")
            sample_rate, data = wavfile.read(j)
            stardata, spec = create_spectrogram(data,L)
            audio_spec={}
            audio_spec['sample_rate']= sample_rate
            audio_spec['spectrogram']= spec
            audio_spec['stardata']= stardata
            audio_spec['audio_data']= data
            validation_spectogram[audio_name]=audio_spec  
    return validation_spectogram

def check_training_model():
    try:
        spectogram_db=pickle.load(open('feature_vectors_spectogram', 'rb'))
        print("Fetched Trained Model")
        return spectogram_db
    except:
        print("Training Model")
        spectogram_db=create_training_model()
        pickle.dump(spectogram_db,open('feature_vectors_spectogram', 'wb'))
        return spectogram_db
    
def check_noise_training_model():
    try:
        spectogram_db=pickle.load(open('feature_vectors_noise_spectogram', 'rb'))
        print("Fetched Trained Model")
        return spectogram_db
    except:
        print("Training Model")
        spectogram_db=create_noise_training_model()
        pickle.dump(spectogram_db,open('feature_vectors_noise_spectogram', 'wb'))
        return spectogram_db

def check_validation_model():
    try:
        validation_spectogram=pickle.load(open('validation_feature_spectogram', 'rb'))
        print("Fetched Validation Model")
        return validation_spectogram
    except:
        print("Training Validation Model")
        validation_spectogram=create_validation_model()
        pickle.dump(validation_spectogram,open('validation_feature_spectogram', 'wb'))
        return validation_spectogram

def to_max_len(arr):
    new_arr=[]
    max_len=0
    for i in arr:
        if(len(i)>max_len):
            max_len=len(i)
    for i in arr:
        if(max_len-len(i)>0):
            app=[]
            for j in range(max_len-len(i)):
                app.append(random.randrange(-50,50)) 
            i=np.append(i,app)
        new_arr.append(i)
    return new_arr


spectogram_db=check_training_model()
spectogram_noise_db=check_noise_training_model()
validation_spectogram=check_validation_model()

training_df=pd.DataFrame.from_dict(spectogram_db,orient='index')
training_noise_df=pd.DataFrame.from_dict(spectogram_noise_db,orient='index')
validation_df=pd.DataFrame.from_dict(validation_spectogram,orient='index')

def process_data(dataframe):
    data=pd.DataFrame(columns=['class_name', 'spec'])
    data_index=dataframe.index
    data_spectograms=[]
    
    for i in range(dataframe.shape[0]):
        data_spectograms.append(dataframe.iloc[i].spectrogram.flatten())
    
    same_sized_spectograms=to_max_len(data_spectograms)
    
    for i in range(dataframe.shape[0]):
        data.loc[i]= [data_index[i].split("/")[0],same_sized_spectograms[i]]
    
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
    x_train.append(features_train.iloc[i]['spec'].flatten())

for i in range(features_train.shape[0]):
    y_train.append(features_train.iloc[i]['class_name'].flatten())

for i in range(features_train.shape[0]):
    x_noise_train.append(features_noise_train.iloc[i]['spec'].flatten())

for i in range(features_train.shape[0]):
    y_noise_train.append(features_noise_train.iloc[i]['class_name'].flatten())

for i in range(features_val.shape[0]):
    x_val.append(features_val.iloc[i]['spec'].flatten())

for i in range(features_val.shape[0]):
    y_val.append(features_val.iloc[i]['class_name'].flatten())

x_train = np. nan_to_num(x_train)
x_noise_train = np. nan_to_num(x_noise_train)
x_val = np. nan_to_num(x_val)

def get_saved_grid():
    try:
        grid=pickle.load(open('spec_grid', 'rb'))
        print("Fetched Saved Grid")
        return grid
    except:
        print("Training Grid ")
        grid = GridSearchCV(SVC(), param_grid, verbose = 3) 
        grid.fit(x_train, y_train)
        pickle.dump(grid,open('spec_grid', 'wb'))
        return grid

def get_noise_saved_grid():
    try:
        grid=pickle.load(open('spec_noise_grid', 'rb'))
        print("Fetched Saved Grid")
        return grid
    except:
        print("Training Grid ")
        grid = GridSearchCV(SVC(), param_grid, verbose = 3) 
        grid.fit(x_noise_train, y_noise_train)
        pickle.dump(grid,open('spec_noise_grid', 'wb'))
        return grid
        
grid=get_saved_grid()
noise_grid=get_noise_saved_grid()

grid_predictions = grid.predict(x_val)
noise_grid_predictions = noise_grid.predict(x_val)
print(classification_report(y_val, grid_predictions,zero_division=0))
print(classification_report(y_val, noise_grid_predictions,zero_division=0))
        
# sample_rate, data = wavfile.read("test.wav")
# stardata, spec = create_spectrogram(data,L)
# plot_spectrogram(spec,sample_rate,L, stardata,data)