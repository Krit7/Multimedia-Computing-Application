#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:50:28 2020

@author: kritverma


REFRENCE TAKEN FROM :- https://github.com/raj1603chdry/CSE3018-Content-Based-Image-and-Video-Retrieval-Lab/tree/master/WEEK4


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

def test_img():
	img=Image.open('test.jpg')
	w,h=img.size
	w=w//4
	h=h//4
	img=img.resize((w,h))
	img_corr=correlogram(img,dist_vector)
	return img_corr
	




#-----------------------------------------------MAIN FUNCTIONS-----------------------------------------------------------
def give_dist_vec():
    dist_vector=[]
    x,y=1,3
    dist_vector.append(x)
    dist_vector.append(y)
    return dist_vector


def get_pos_neg(good,ok,junk,retrieved,db_corr):
    true_pos=0
    false_neg=0
    track=0
    
    arr=np.append(good,ok)
    arr_check=np.append(arr,junk)
    tot_img=len(arr_check)

    for j in retrieved:
        if(track<tot_img):
            name=db_corr.iloc[j[0]]
            i_name=name.name
            if(i_name in arr_check):
                true_pos+=1
            elif(i_name not in arr_check):
                false_neg+=1
        else:
            break
        track+=1

    return (true_pos,false_neg)
    


def y_coordinate(n,y,Y):
    n_len=8*n

    n_y=np.zeros([n_len])

    
    n_y[0]=y
    d=1
    start=1
    stop=n+1
    while(start<stop):
        n_y[start]=y-d
        d=d+1
        start+=1
    n_y[n:3*n+1:1]=y-n;
    
    d=0
    start=3*n+1
    stop=5*n+1
    while(start<stop):
        n_y[start]=y-n+d
        d=d+1
        start+=1
    n_y[5*n+1:7*n+1:1]=y+n
    
    d=0;
    start=7*n+1
    stop=7*n+1+(n-1)
    step=1
    while(start<stop):
       n_y[start]=y+n-d;
       d=d+1;
       start+=step
     
    return n_y

def x_coordinate(n,x,X):
    n_len=8*n
    n_x=np.zeros([n_len])
    
    n_x[0]=x-n;
    
    n_x[1:1+n:1]=x-n;
    
    d=0;
    start=1+n
    stop=3*n+1
    step=1
    while(start<stop):
        n_x[start]=x-n+d;
        d=d+1;
        start+=step
    n_x[3*n+1:5*n+1]=x+n;
    
    d=0;
    start=5*n+1
    stop=7*n+1
    while(start<stop):
        n_x[start]=x+n-d;
        d=d+1;
        start+=1
    n_x[7*n+1:7*n+1+(n-1)]=x-n;
     
    return n_x

def get_n(n,x,y,color,quant_img,X,Y):
    len_n=8*n
    valid_vector=np.zeros([len_n])
    pos_count=0
    tot_count=0
    
    nbrs_x=x_coordinate(n,x,X)
    nbrs_y=y_coordinate(n,y,Y)

    for i in range(len_n):
        
        x_true=nbrs_x[i]>0 and nbrs_x[i]<=X
        y_true=nbrs_y[i]>0 and nbrs_y[i]<=Y
        
        if ( x_true and y_true):    
            valid_vector[i]=1
        else:
            valid_vector[i]=0
    
    for i in range(len_n):
       if (valid_vector[i]==1):
           coordinate=(nbrs_y[i]-1,nbrs_x[i]-1)
           data=quant_img.getpixel(coordinate)
           if (data==color):
               pos_count=pos_count+1
        
       tot_count=tot_count+1; 
    
    return (pos_count,tot_count)


def correlogram(img,dist_vector):
    correlogram_vector=[];
    
    gray_img=img.convert('LA')
    
    Y,X=gray_img.size
    
    quant_img=img.quantize(256)
    
    d=len(dist_vector)
    
    count_matrix=np.zeros([256,d])  
    total_matrix=np.zeros([256,d])
    prob_dist=np.ndarray(shape=d,dtype=list)
    
    for i in range(d):
        for x in range(X):
            for y in range(Y):
                coordinate=(y,x)
                color=quant_img.getpixel(coordinate)
                
                pos_count,tot_count=get_n(dist_vector[i],x,y,color,quant_img,X,Y)
                count_matrix[color,i]+=pos_count
                total_matrix[color,i]+=tot_count
                
        prob_dist[i]=np.divide(count_matrix[:,i],total_matrix[:,i])
    
    for i in range(d):
        correlogram_vector=np.concatenate((correlogram_vector,prob_dist[i]))
    return correlogram_vector

def model_db_corr(dist_vector):
    count=0
    corr_db={}
    for i in glob.glob('../images/*.jpg'):
        curr=time.time()
        count+=1
        name=i.split('/',2)[2]
        img=Image.open(i)
        w,h=img.size
        w=w//4
        h=h//4
        img=img.resize((w,h))
        img_corr=correlogram(img,dist_vector)
        corr_db[name]=img_corr
        end=time.time()
        print(count,end-curr)
        
    db_corr_df=pd.DataFrame.from_dict(corr_db,orient='index')
    return db_corr_df


def check_for_model(dist_vector):
    try:
        db_corr=pickle.load(open('feature_vectors_cac', 'rb'))
        print("Fetched Trained Model")
        
        sr=db_corr.index
        new_index=[]
        for i in sr:
            new_name=i.split("/")[4]
            new_index.append(new_name)
        
        db_corr.index=new_index
        
        return db_corr
    except:
        db_corr=model_db_corr(dist_vector)
        print("Training Model")
        return db_corr


def input_stream(db_corr):
    all_prec=[]
    avg_good_count=0
    avg_ok_count=0
    avg_junk_count=0
    
    # ---------------------------------------Reading Query Images----------------------------------------------
    for i in train_query:
        rel_corr_db={}
        query_file_name=i.split("/",3)[3]
        query_file_name=query_file_name.replace("_query.txt","")
        
        with open(i, 'r') as f:
            query_data = f.read()
            query_img_name = query_data.split(" ",1)[0][5:]+'.jpg'
            query_img_corr=db_corr.loc[query_img_name,:]
        
        for j in range(len(db_corr)):
            img_corr=db_corr.iloc[j,:]
            dist=df.sum(abs(query_img_corr-img_corr))
            rel_corr_db[j]=dist
        
        sorted_rel_corr = sorted(rel_corr_db.items(), key=operator.itemgetter(1))
        good,ok,junk=[],[],[]
        g_count,o_count,j_count=0,0,0
        tot_img=0
        
        # ------------------------------------Fetching Ground Truths ---------------------------------------------
        for j in train_ground:
            file_name=j.split("/",3)[3]
            
            if(query_file_name in file_name):
                
                if('good' in file_name):
                    with open(j, 'r') as f:
                        line_count=0
                        for line in f:
                            line=line.replace("\n",".jpg")
                            good.append(line)
                            g_count+=1
                            line_count+=1
                        tot_img+=line_count
                
                elif('junk' in file_name):
                    with open(j, 'r') as f:
                        line_count=0
                        for line in f:
                            line=line.replace("\n",".jpg")
                            junk.append(line)
                            j_count+=1
                            line_count+=1
                        tot_img+=line_count
                
                else:
                    with open(j, 'r') as f:
                        line_count=0
                        for line in f:
                            line=line.replace("\n",".jpg")
                            ok.append(line)
                            o_count+=1
                            line_count+=1
                        tot_img+=line_count
        
        # -------------------------------Calculating Precision & Recall ------------------------------------
        
        true_pos,false_neg=get_pos_neg(good,ok,junk,sorted_rel_corr,db_corr)
        prec=true_pos/(true_pos+false_neg)
        all_prec.append(prec)
        
        # -------------------------------Calculating Avg Good,Ok,Junk ------------------------------------
        act_g_count,act_o_count,act_j_count=0,0,0
        track=0
        for j in sorted_rel_corr:
            if(track<=tot_img):
                name=db_corr.iloc[j[0]]
                i_name=name.name
                if(i_name in good):
                    act_g_count+=1
                elif(i_name in junk):
                    act_j_count+=1
                elif(i_name in ok):
                    act_o_count+=1
            else:
                break
            track+=1
        avg_good_count+=act_g_count/g_count
        avg_ok_count+=act_o_count/o_count
        avg_junk_count+=act_j_count/j_count
    
    max_prec=max(all_prec)*100
    min_prec=min(all_prec)*100
    avg_prec=(sum(all_prec)/len(all_prec))*100
    
    
    
    print("Maximum Precision", "{0:.2f}".format(max_prec))
    print("Minimum Precision", "{0:.2f}".format(min_prec))
    print("Average Precision", "{0:.2f}".format(avg_prec))
    
    print("Average Good Retrieved", round(avg_good_count,2))
    print("Average Ok Retrieved", round(avg_ok_count,2))
    print("Average Junk Retrieved", round(avg_junk_count,2))
    
    print("*NOTE:- ALL VALUES OF PRECISION, RECALL AND F1 ARE SAME")
#--------------------------------------------------MAIN CALL----------------------------------------------------------
dist_vector=give_dist_vec()
test_img_corr=test_img()
# db_corr=check_for_model(dist_vector)
# input_stream(db_corr)