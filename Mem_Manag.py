import load_data
import os
i_train_path = "MEG_data/Final Project data/Intra/train"
i_test_path = "MEG_data/Final Project data/Intra/test"

c_train_path = "MEG_data/Final Project data/Cross/train"
c_test1_path = "MEG_data/Final Project data/Cross/test1"
c_test2_path = "MEG_data/Final Project data/Cross/test2"
c_test3_path = "MEG_data/Final Project data/Cross/test3"

def fit_eight(dirlist, parentpath): #Fit a network on 8 datafiles at a time.
    fitlist = [] 
    i = 0
    while i < 8:    #Load the 8 datafiles
        fitlist[i] = load_data.read_data_file(parentpath + "/" + dirlist[i])

    for i in fitlist:   #Fit the network on the datafiles
        #nn.fit()
        NotImplemented
    

def train_dir(dirpath):   #Train a network on a given directory
    dirnames = os.listdir(dirpath)  #get list of all filenames (IF THERE ARE DIRECTORIES NEXT TO FILES THIS CRASHES BUT THEY SHOULDN'T BE THERE)
    batches = len(dirnames)//8  #int divide by eight to avoid float errors
    if not (batches % 8) == 0:  #Check if it is Divisible by eight (This should be the case for all folders! Might need a workaround with downsampling idk)
        raise TypeError("Not divisible by eight")
    i = 0 
    while i < batches:  #Fit on 8 batches at a time
        fit_eight(dirnames[i*8:(i+1)*8], dirpath)

    






    