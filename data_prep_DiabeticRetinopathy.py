# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 19:16:16 2024

@author: Rajesh
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
import matplotlib.cm as cm
import cv2
import umap
#import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
#from PIL import Image
#import scipy

image_dir = Path('D:/Work/Kaggle/DiabeticRetinopathy/colored_images')

# Get filepaths and labels
filepaths = list(image_dir.glob(r'**/*.png'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

df = pd.read_csv("D:/Work/Kaggle/DiabeticRetinopathy/train.csv")
df

filepaths

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate filepaths and labels
image_df = pd.concat([filepaths, labels], axis=1)

# Shuffle the DataFrame and reset index
image_df = image_df.sample(frac=1).reset_index(drop = True)

# Show the result
image_df.head()

image_df

level = []
for i in image_df['Label']:
    if i=='No_DR':
        level.append(0)
    elif i=='Mild':
        level.append(1)
    elif i=='Moderate':
        level.append(1)
    elif i=='Severe':
        level.append(1)
    else:
        level.append(1)
        
image_df['Level'] = level
image_df.head()

X = [];#np.empty((0, 0)) 
ik = 0;
for i in image_df['Filepath']:
    image = cv2.imread(i)
    
   
    
    temp = np.reshape(image,-1);
    
    #X = np.stack((X, temp), axis=1);
    
    #X[:,ik] = temp;
    if ik == 0:
      X = temp;
    
    elif ik == 1:
        
        X = np.stack((X,temp),axis=1);
        X = np.transpose(X);
        
        
    else:
        
        #X[:,ik-1] = temp;# np.append(X,temp,axis=1); 
        X = np.vstack((X,temp));
      
    ik = ik + 1;      
        
#    X.append(image)
    
#X = np.asarray(X)
y = image_df['Level']
Y = np.asarray(y)
image.shape


scaler = preprocessing.StandardScaler().fit(X)
scaler

scaler.mean_

scaler.scale_

X_scaled = scaler.transform(X)

trans = umap.UMAP(n_neighbors=15,min_dist=0.1,n_components=4).fit(X_scaled)
X_transformed = trans.transform(X_scaled);

np.savetxt("DiabeticRetinopathy_normalized_4_python.csv", X_transformed, delimiter=",");
np.savetxt("DiabeticRetinopathy_data_label.csv", Y, delimiter=",");

