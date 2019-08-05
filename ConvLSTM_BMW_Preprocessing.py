import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re
import cv2 as cv
import scipy
from multiprocessing.dummy import Pool as ThreadPool

import os, math, timeit, json, csv
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt



FULLPATHTOCOMPILEDCSVFILE = "C:\\Users\\dhanover\\Documents\\AI-Repo-master\\Autopilot-TensorFlow-master\\Autopilot-TensorFlow-master\\data.txt"
CV_LOAD_IMAGE_GRAYSCALE = 0 # read in images as greyscale

###########################################################
#Get GreyScale Images first, then calculate optical flow
###########################################################

def import_data(FULLPATHTOCOMPILEDCSVFILE):
    """
    Info:
    -----------------
    Import data.txt with image and steering angle meta data
    Return list of images and steering angles as array
    """
    with open(FULLPATHTOCOMPILEDCSVFILE) as f:
        xs = []
        ys = []
        for line in f:
            xs.append("preprocessed_data/data/" + re.split(',| ',line)[0])
            ys.append(float(re.split(',| ',line)[1]))
    return xs, ys 

def threaded_preprocess_data(X):
    """
    Info:
    -----------------
    Make data ready for NN fitting

    Steps:
    -----------------
    - Resize images to 200x66x1 (greyscale)
    """
    x_resized = cv.resize(cv.imread(X, CV_LOAD_IMAGE_GRAYSCALE), (200,66)) #import images as greyscale and resize to 200x66
    cv.imwrite(f"C:\\Users\\dhanover\\Documents\\AI-Repo-master\\Autopilot-TensorFlow-master\\Autopilot-TensorFlow-master\\preprocessed_data\\{X}", x_resized)
    
def preprocess_data(X,y):
    """
    Info:
    -----------------
    Try and load images into memory, calc optical flow, then scale the data sets for both images and optical flow seperately

    Steps:
    -----------------
    - Calculate bi-directional optical flow
    - Normalize the data set (3@200x66)
    """
    flag = True
    images = [] #init palceholder for all images to be loaded into mem
    for i in X:
        images.append(cv.imread(i, CV_LOAD_IMAGE_GRAYSCALE))
    prev = images[0]
    flow = [] #init var for optical flow calcs
    count = 1
   
    for k in images[1:]:
        flow=cv.calcOpticalFlowFarneback(prev,k, None, 0.5, 3, 15, 3 , 5, 1.2, 0)
        horz = cv.normalize(flow[...,0], None, 0, 255, cv.NORM_MINMAX) #motion to the extreme left is black, motion to extreme white is 255
        vert = cv.normalize(flow[...,1], None, 0, 255, cv.NORM_MINMAX) 
        horz = horz.astype('uint8')
        vert = vert.astype('uint8') 
        cv.imwrite(f"C:\\Users\\dhanover\\Documents\\AI-Repo-master\\Autopilot-TensorFlow-master\\Autopilot-TensorFlow-master\\preprocessed_data\\horizontal_flow\\{count}.jpg",horz)
        cv.imwrite(f"C:\\Users\\dhanover\\Documents\\AI-Repo-master\\Autopilot-TensorFlow-master\\Autopilot-TensorFlow-master\\preprocessed_data\\vertical_flow\\{count}.jpg",vert)
        count += 1
        prev = k

if __name__ == '__main__':

    X, Y = import_data(FULLPATHTOCOMPILEDCSVFILE)
    #pool = ThreadPool()
    
    #pool.map(threaded_preprocess_data,X)
    #pool.close()
    #pool.join()

    #Can't run optical flow calcs in parallel because need to pass in adjacent frames
    preprocess_data(X,Y)
