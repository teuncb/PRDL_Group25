import load_data
import downsampling
import re
import os
import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


i_train_path = "MEG_data/Final Project data/Intra/train"
i_train_prepro_path = "MEG_data/Final Project data/Intra/train_prepro"
i_test_path = "MEG_data/Final Project data/Intra/test"

c_train_path = "MEG_data/Final Project data/Cross/train"
c_train_prepro_path = "MEG_data/Final Project data/Cross/train_prepro"
c_test1_path = "MEG_data/Final Project data/Cross/test1"
c_test2_path = "MEG_data/Final Project data/Cross/test2"
c_test3_path = "MEG_data/Final Project data/Cross/test3"

def fit_eight(dirlist, parentpath, epochs):                                 #Fit a network on 8 datafiles at a time.
    fitlist = [] 
    i = 0
    j = 0
    while i < 8:                                                    #Load the 8 datafiles
        filename = parentpath + "/" + dirlist[i]            
        data = load_data.read_data_file(filename)
        label = extract_label(filename)
        #Scale
        fitlist[i] = (data, label)

    while j < epochs:
        for i in fitlist:                                               #Fit the network on the datafiles
            #nn.fit()
            NotImplemented
    

def train_dir(dirpath, epochs, shuffle=True):                                             #Train a network on a given directory
    dirnames = os.listdir(dirpath)                                  #get list of all filenames (IF THERE ARE DIRECTORIES NEXT TO FILES THIS CRASHES BUT THEY SHOULDN'T BE THERE)
    batches = len(dirnames)//8                                      #int divide by eight to avoid float errors
    if not (batches % 8) == 0:                                      #Check if it is Divisible by eight (This should be the case for all folders! Might need a workaround with downsampling idk)
        raise TypeError("Not divisible by eight")
    i = 0 
    while i < batches:                                              #Fit on 8 batches at a time
        fit_eight(dirnames[i*8:(i+1)*8], dirpath, epochs)

def extract_label(filename):                                        #Extract the label out of a filename
    filename = filename.split("/")
    pattern = r'_\d'
    split = re.split(pattern, filename[-1])
    return split[0]


def prepro_cross_files():                                          #Preprocess all the files and save them
    dirnames = os.listdir(c_train_path)                                 
    for dir in dirnames:
        data = load_data.read_data_file(c_train_path + "/" + dir)
        new_data = downsampling.downsample_matrix(data, 3)
        scaled_data = load_data.scale(new_data, StandardScaler(), timewise=True)      #Beg that this returns an nparray
        try:
            hfive = h5py.File(c_train_prepro_path + "/" + dir, 'w')
        except:
            raise Exception("You forgot to make a train_prepro path")              #Can maybe be automated idk
        hfive.create_dataset('dir', data=scaled_data)
        hfive.close()

def prepro_intra_files():                                          #Preprocess all the files and save them
    dirnames = os.listdir(i_train_path)                            #Get all the original files  
    for dir in dirnames:                                            #For each file
        data = load_data.read_data_file(i_train_path + "/" + dir)                       #Load the data
        new_data = downsampling.downsample_matrix(data, 3)                              #Downsample the data
        scaled_data = load_data.scale(new_data, StandardScaler(), timewise=True)        #Scale the data
        try:
            hfive = h5py.File(c_train_prepro_path + "/" + dir, 'w')
        except:
            raise Exception("You forgot to make a train_prepro path")                       #Open a new h5 object
        hfive.create_dataset('dir', data=scaled_data)                                   #Enter data into the object
        hfive.close()                                                                   #Close (save) the object




#print("pre")
#prepro_cross_files()
#print("pri")
#prepro_intra_files()
#print("pro")

test = load_data.read_data_file("MEG_data/Final Project data/Intra/train/rest_105923_1.h5")
print(test.shape)
test_prepro = load_data.read_prepro_file("MEG_data/Final Project data/Intra/train_prepro/rest_105923_1.h5")
print(test_prepro.shape)







    