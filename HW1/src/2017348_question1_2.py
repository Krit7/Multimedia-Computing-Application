#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:12:24 2020

@author: kritverma


Refrence Has Been Taken From :- https://projectsflix.com/opencv/laplacian-blob-detector-using-python/

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt 
from scipy import spatial
import math

import time
import glob
import pandas as pd
import pickle


result_images_path='./results/'
train_query=glob.glob('../train/query/*.txt')

k = 1.414
sigma = 1.0
pi=np.pi

# -------------------------------------------HELPER FUNCTIONS-------------------------------------------------
def get_square(x):
    return np.square(x)

def add_row_padding(arr):
  row_len=len(arr[0])
  row_padding=np.zeros(row_len,dtype='int32')
  front_padding=np.vstack((row_padding,arr))
  end_padding=np.vstack((front_padding,row_padding))
  
  return end_padding

def add_col_padding(arr):
  col_len=len(arr)
  col_padding=np.zeros(col_len,dtype='int32')
  front_padding=np.column_stack((col_padding,arr))
  end_padding=np.column_stack((front_padding,col_padding))
  
  return end_padding

def add_padding(arr):
  row_padding=add_row_padding(arr)
  col_padding=add_col_padding(row_padding)
  
  return col_padding

def limit_range(x,lim_range):
    low=lim_range[0]
    high=lim_range[1]
    if(x>high):
      return high
    elif(x<low):
      return low
    else:
      return x


def generate_var(r1,r2,d):
    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    
    return (a,b,c,d)

def calc_area_blobs(r1,r2,d):
    var_dist=generate_var(r1,r2,d)
    lim_range=(-1,1)
    r1_sq = r1 ** 2
    r2_sq = r2 ** 2
    d_sq = d ** 2
    
    ratio1 = (d_sq + r1_sq - r2_sq) / (2 * d * r1)
    ratio1 = limit_range(ratio1,lim_range)
    acos1 = math.acos(ratio1)
    
    ratio2 = (d_sq + r2_sq - r1_sq) / (2 * d * r2)
    ratio2 = limit_range(ratio2,lim_range)
    acos2 = math.acos(ratio2)
    
    r1_cos = r1_sq*acos1
    r2_cos = r2_sq*acos2
    abs_dist=np.sqrt(abs(var_dist[0] * var_dist[1] * var_dist[2] * var_dist[3]))
    area = (r1_cos + r2_cos -0.5 * abs_dist)
    
    return area
    
def get_final_filter(x,y,x_filter,y_filter):
	final_filter = (-(2*sigma**2) + (x*x + y*y) ) * (x_filter*y_filter) * (1/(2*np.pi*sigma**4) )
	return final_filter

def get_slice_val(i,j):
    s_list=[]
    s_list.append(i-1)
    s_list.append(i+2)
    s_list.append(j-1)
    s_list.append(j+2)
    
    return s_list     

# --------------------------------------------------MAIN FUNCTIONS--------------------------------------------
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
    
    gen_filter=(-(2*sigma**2) + (x*x + y*y) ) * (x_filter*y_filter) * (1/(2*np.pi*sigma**4) )
    return gen_filter

def LoG_convolve(img):
    i=0
    log_images = []
    no_conv=9
    while i<no_conv:
        y = k**i 
        sigma_1 = sigma*y
        filter_log = LoG(sigma_1)
        
        image = cv2.filter2D(img,-1,filter_log) # convolving image
        image=add_padding(image)
        image = np.square(image) 
        log_images.append(image)
        i+=1
    log_image_np = np.asarray(log_images) 
    return log_image_np

def detect_blob(log_image_np,img):
    co_ordinates = []
    threshold=0.03
    h = img.shape[0]
    w = img.shape[1]
    i,j=1,1
    for i in range(1,h):
        for j in range(1,w):
            slicing_list=get_slice_val(i,j)
            slice_img = log_image_np[:,slicing_list[0]:slicing_list[1],slicing_list[2]:slicing_list[3]] 
            result = np.amax(slice_img) 
            if result < threshold:
                continue
            elif result >= threshold:
                h_s = slice_img.shape[0]
                w_s = slice_img.shape[1]
                cc=np.unravel_index(slice_img.argmax(),slice_img.shape)
                z = cc[0]
                x = cc[1]
                y = cc[2]
                r=(k**z)*sigma
                co_ordinates.append((i+x-1,j+y-1,r))
    return co_ordinates

def blob_overlap(blob1, blob2):
    n_dim = 2
    root_ndim = 1.414
    
    r1 = blob1[2] * 1.414
    r2 = blob2[2] * 1.414
    
    blob1_xy=blob1[:2]
    blob2_xy=blob2[:2]
    
    d = np.sqrt(np.sum(( blob1_xy - blob2_xy )**2))
    
    r1_r2_sum=np.add(r1,r2)
    r1_r2_sub=np.subtract(r1,r2)
    
    if d <= abs(r1_r2_sub):
        return 1
    
    elif d > r1_r2_sum:
        return 0
    
    else:
        area=calc_area_blobs(r1,r2,d)
        min_r=min(r1, r2)
        area/=(pi * (np.square(min_r)))
        return area

def redundancy(blobs_array, overlap):
    sigma = blobs_array[:, -1].max()
    distance = 2 * sigma * np.sqrt(blobs_array.shape[1] - 1)
    tree = spatial.cKDTree(blobs_array[:, :-1])
    pairs = np.array(list(tree.query_pairs(distance)))
    
    if len(pairs) == 0:
        return blobs_array
    else:
        for (i, j) in pairs:
            blob1 = blobs_array[i]
            blob2 = blobs_array[j]
            check_overlap=blob_overlap(blob1, blob2)
            if check_overlap > overlap:
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0
    filtered_arr=[]
    for b in blobs_array:
        if b[-1] > 0:
            filtered_arr.append(b)
    
    return np.asarray(filtered_arr)


# -------------------------------------------CALLING FUNCTIONS-------------------------------------------------
def get_query_img():
    track=0
    query_images={}
    for i in train_query:
        if(track<5):
            query_file_name=i.split("/",3)[3]
            query_file_name=query_file_name.replace("_query.txt","")
            with open(i, 'r') as f:
                query_data = f.read()
                query_img_name = query_data.split(" ",1)[0][5:]+'.jpg'
                img_path='../images/'+query_img_name
                img=cv2.imread(img_path,0)
                img=cv2.resize(img,(300,300))
                img = img/255.0 
                query_images[query_img_name]=img
            track+=1
        else:
            break
    return query_images


def model_query_blob(query_images):
    query_images_blob={}
    for name,img in query_images.items():
        curr=time.time()
        log_image_np = LoG_convolve(img)
        co_ordinates = list(set(detect_blob(log_image_np,img)))
        co_ordinates = np.array(co_ordinates)
        co_ordinates = redundancy(co_ordinates,0.5)
        query_images_blob[name]=co_ordinates
        end=time.time()
        print(name,end-curr)
    
    query_img_blobs=pd.DataFrame.from_dict(query_images_blob,orient='index')
    return query_img_blobs

def check_for_model(query_images):
    try:
        query_images_blob=pickle.load(open('feature_vectors_blob', 'rb'))
        print("Fetched Trained Model")
        return query_images_blob
    except:
        print("Training Model")
        query_images_blob=model_query_blob(query_images)
        pickle.dump(query_images_blob,open('feature_vectors_blob', 'wb'))
        return query_images_blob


# -------------------------------------------MAIN CALL-------------------------------------------------
query_images=get_query_img()
query_images_blob=check_for_model(query_images)

for i,j in query_images_blob.iterrows():
    img=query_images.get(i)
    fig, ax = plt.subplots()
    nh = img.shape[0]
    nw = img.shape[1]
    ax.imshow(img, interpolation='nearest',cmap="gray")
    
    for blob_row in j:
        for blob in blob_row:
            y,x,r = blob
            mark_coordinate=(x, y)
            r*=1.414
            c = plt.Circle(mark_coordinate, r, color='red', linewidth=1, fill=False)
            ax.add_patch(c)
        ax.plot()
        plt.title(i)
        plt.show()





    






