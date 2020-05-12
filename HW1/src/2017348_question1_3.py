#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:50:28 2020

@author: kritverma


"""
#-------------------------------------------------IMPORTS-----------------------------------------------------------
from PIL import Image
import numpy as np
import glob
import time
import pandas as pd
from pandas import DataFrame as df
import pickle
import operator

np.seterr(divide='ignore', invalid='ignore')

#--------------------------------------------------PATHS-----------------------------------------------------------
train_query=glob.glob('../train/query/*.txt')
train_ground=glob.glob('../train/ground_truth/*.txt')


#----------------------------------------------TESTING FUNCTIONS-----------------------------------------------------------

def generate_filter(f,sigma):
    sigma_sq=2.0*(sigma**2)
    f_sq=get_square(f)
    filter_power=f_sq/sigma_sq
    # print("Called 1")
    return np.exp(-filter_power)

def LoG(sigma):
    sigma_sq=2.0*(sigma**2)
    n = np.ceil(sigma*6)
    
    m_size=n//2
    r=-1*m_size
    c=m_size+1
    
    y,x = np.ogrid[r:c,r:c]
    y_filter = generate_filter(y,sigma)
    x_filter = generate_filter(x,sigma)
    
    x_sq=get_square(x)
    y_sq=get_square(y)
    
    final_filter = (-(2*sigma**2) + (x*x + y*y) ) *  (x_filter*y_filter) * (1/(2*np.pi*sigma**4))
    # print("Called 2")
    return final_filter

def LoG_convolve(img):
    log_images = []
    for i in range(9):
        y = k**i 
        sigma_1 = sigma*y
        filter_log = LoG(sigma_1)
        
        image = cv2.filter2D(img,-1,filter_log) # convolving image
        image=add_padding(image)
        image = np.square(image) 
        log_images.append(image)
    
    # log_image_np = np.array([i for i in log_images])
    log_image_np = np.asarray(log_images) 
    return log_image_np