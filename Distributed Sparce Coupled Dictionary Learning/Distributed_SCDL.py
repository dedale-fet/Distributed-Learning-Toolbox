"""Distributed Sparse Coupled Dictionary Learning

The main script for execution

:Author: Nancy Panousopoulou@FORTH-ICS <apanouso@ics.forth.gr>

:Reference documents: 

 [1] K.  Fotiadou, G. Tsagkatakis, P. Tsakalides `` Linear Inverse Problems with Sparsity Constraints,'' DEDALE DELIVERABLE 3.1, 2016.
 
 [2] K. Fotiadou, G. Tsagkatakis, P. Tsakalides. Spectral Resolution Enhancement via Coupled Dictionary Learning. 2017. Under Review in IEEE Transactions on Remote Sensing.

:Date: 01/02/2018

"""

import numpy as np
import scipy.io
import os.path
from pyspark import SparkContext
from pyspark.sql import * 



from subprocess import call
import time

from CDLOps import *
from numpy import linalg as la

import math
from numpy import genfromtxt



import scipy.io as sio
from numpy.linalg import inv


from args import get_opts

###########################################################################
def calcSum2(data1, data2):
    """
    Method for calculating the error vector over the cluster (reduce).
    
    This method should follow the calcErr function for completing the error calculation. 

    Input Arguments
    ----------
    data1: RDD data block (np.ndarray(2D), np.ndarray(2D))
           input tuple of the high resolution data (RDD blocks) 
    
    data2: RDD data block (np.ndarray(2D), np.ndarray(2D))
           input tuple of the high resolution data (RDD blocks)
           
    Returns:
    ---------
    err1: nparray
          the error vector for the high-resolution dictionary
    
    err2: nparray 
         the error vector for the high-resolution dictionary
    """ 
    
    data11 = data1[0]
    data12 = data1[1]
    
    data21 = data2[0]
    data22 = data2[1]
    
    
    err1 = [data11[i] + data21[i] for i in range(0,len(data11))]
    err2 = [data12[i] + data22[i] for i in range(0,len(data12))]
    
    
    return [err1, err2]

#######################################################################
def calcErr(dicth,dictl,cdli):
    
    """
    Method for calculating the error vector over the cluster (map).
    
    This method should be followed by the calcSum2 function for completing the error calculation. 

    Input Arguments
    ----------
    dicth: np.ndarray (2D)
           the high-resolution dictionary 
    
    dictl: np.ndarray (2D)
           the low-resolution dictionary
           
    cdli: RDD CDL object
          the updated CDL object
           
    Returns:
    ---------
    err1: nparray
          the error vector for the high-resolution dictionary
    
    err2: nparray 
         the error vector for the high-resolution dictionary
    """ 
    
            
    errh = np.square((cdli.datain_h - np.dot(dicth, np.transpose(cdli.wh))))
    errl = np.square((cdli.datain_l - np.dot(dictl, np.transpose(cdli.wl))))
    
         
    return [errh, errl]

#########################################################################################


#########################################################################################
def getVals(cdli):
    
    """
    Method for calculating the [(S_h\timesW_h), phi_h] and [(S_l\timesW_l), phi_l] the over the cluster (Outer product - map).
    
    This method should be followed by calcSum3 for completing the distributed matrices calculation.

    Input Arguments
    ----------
    cdli: RDD CDL object
          the updated CDL object
           
    Returns:
    ---------
    swh: np.ndarray
          the $S \times W$ matrix for the high-resolution dictionary
          
    swl: np.ndarray
          the $S \times W$ matrix for the low-resolution dictionary
    
    phih: nparray
          the phi vector for the high-resolution dictionary
    
    phil: nparray 
         the phi vector for the low-resolution dictionary
    """
    
    
    swh = calcOutProd(np.reshape(cdli.datain_h, (cdli.datain_h.shape[0],1)), np.reshape(cdli.wh, (1, cdli.wh.shape[0])))
    swl = calcOutProd(np.reshape(cdli.datain_l, (cdli.datain_l.shape[0],1)), np.reshape(cdli.wl, (1, cdli.wl.shape[0])))        
    phi_h = [cdli.wh[i]*cdli.wh[i] for i in range(0,len(cdli.wh))] 
    phi_l = [cdli.wl[i]*cdli.wl[i] for i in range(0,len(cdli.wl))]
    
    return [[swh, swl], [phi_h, phi_l]]

##################################################################

def calcSum3(data1, data2):
    """
    Method for calculating the [(S_h\timesW_h), phi_h] and [(S_l\timesW_l), phi_l] the over the cluster (Outer product - reduce)
    
    This method should follow the getVals method for the distributed matrices calculation.
    Input Arguments
    ----------
    data1: RDD data block (np.ndarray(2D), np.ndarray(2D), np.ndarray(2D), np.ndarray(2D))
           input tuple of the high resolution data (RDD blocks) 
    
    data2: RDD data block (np.ndarray(2D), np.ndarray(2D),np.ndarray(2D), np.ndarray(2D))
           input tuple of the high resolution data (RDD blocks)
           
    Returns:
    ---------
    swh: np.ndarray
          the $S \times W$ matrix for the high-resolution dictionary
          
    swl: np.ndarray
          the $S \times W$ matrix for the low-resolution dictionary
    
    phih: nparray
          the phi vector for the high-resolution dictionary
    
    phil: nparray 
         the phi vector for the low-resolution dictionary
    """
       
    #SxW @high resolution
    data111 = data1[0][0]
    #SxW @low resolution
    data112 = data1[0][1]
    #phi @high resolution
    data121 = data1[1][0]
    #phi @low resolution
    data122 = data1[1][1]
    
    #SxW @high resolution
    data211 = data2[0][0]
    #SxW @low resolution
    data212 = data2[0][1]
    #phi @high resolution
    data221 = data2[1][0]
    #phi @low resolution
    data222 = data2[1][1]
    
    
      
    swh = [data111[i] + data211[i] for i in range(0,len(data111))]
    swl = [data112[i] + data212[i] for i in range(0,len(data112))]
    
    phih = [data121[i] + data221[i] for i in range(0,len(data121))]
    phil = [data122[i] + data222[i] for i in range(0,len(data122))]
    
    
    return [[swh, swl], [phih, phil]]
#################################################################################
    

################################################################################   
def normD(dictin):
    """
    Normalize the dictionary between [0,1] 
    
    Input Arguments
    ----------
    dictin : np.ndarray (2D)
        The input dictionary
        
        
    Returns:
    the normalized dictionary (2D)   
    """
    
    
    tmp = 1 / np.sqrt(np.sum(np.multiply(dictin, dictin), axis=0))
    return np.dot(dictin, np.diag(tmp))

#################################################################################

def run_script(sc):
    """
    The main execution script
    
    Input Arguments
    ----------
    sc : Spark Context Object
        The spark context object containing the primary spark configuration parameters
        
        
    Returns:
    1 (all results are stored in a designated mat file)
    """
    
    
    
    
    
    
    #the size of the image
    imageN = opts.imageN
    #the size of the dictionary
    dictsize = opts.dictsize
    #the number of bands in high resolution
    bands_h_N = opts.bands_h
    
    #the number of bands in low resolution
    bands_l_N = opts.bands_l
    
    
    #parameters for training. 
    c1 = opts.c1 # Default value: 0.4
    
    c2 = opts.c2 # Default value: 0.4
    
    c3 = opts.c3 # Default value: 0.8
    
    maxbeta = opts.maxbeta #Default value: 1e+6
    
    delta = opts.delta #Default value: 1e-4
    beta = opts.beta #Default value: 0.01
    
    #The learning rate thresholding value
    lamda = opts.lamda
    
    #number of iterations for training
    train_iter =opts.n_iter
    
    #the number of partitions for the data parallelization into RDD blocks
    partitions_N = opts.partitions
    
    
        
    #input data are in the from (# of samples) x (# of Bands)
    #high resolution samples
    data_h = sc.parallelize(genfromtxt(opts.inputhigh, delimiter=','), partitions_N).cache()
    
    #low resolution samples
    data_l = sc.parallelize(genfromtxt(opts.inputlow, delimiter=','), partitions_N).cache()

     
    #initializing the dictionary @ high resolution
    dict_h_t = data_h.take(dictsize)
    
    #initializing the dictionary @ low resolution
    dict_l_t = data_l.take(dictsize)
   
    
    dict_h = np.transpose(dict_h_t)
    dict_l = np.transpose(dict_l_t)
    
    print('>>>>>>>>>>>>>>>>')
    print(data_h.count())
    print(data_l.count())
    
    #bundling the input samples together
    datain = data_h.zip(data_l).cache()
    print('>>>>>>>>>>>>>>>>')
   
    print(dict_h.shape)
    print(dict_l.shape)
    
    #optional: uncomment to save the initial values of the dictionaries
    #mat2save =  './tttmpinit' + '_' + str(imageN) + 'x' + str(dictsize) + '.mat'  
    #sio.savemat(mat2save, {'dicth_init':dict_h, 'dictl_init': dict_l})
   
   
    #operational parameters to be broadcasted to the cluster
    lamda_bc = sc.broadcast(lamda)
    dictsize_bc = sc.broadcast(dictsize)
    
    
    #broadcast the dictionaries....
    dict_h_bc = sc.broadcast(dict_h)
    dict_l_bc = sc.broadcast(dict_l)
    
    
    #define and initialize the CDL object for optimization.
    tmp = datain.map(lambda x: startCDL(x, dictsize_bc.value)).cache() 
    
    
  
    wind = opts.window;    
    
    #######################################
    ##initialize the variables for error convergence checking...
    err_h = 0;
    err_l = 0;
    
    buff_size = 10 #to do - put this in arg in...
    err_thrs = 0.01 #likewise
    
    m = 0
    mm = 0
    time_all = []	
    ##keep the error buffer size as an even number.
    if buff_size % 2 == 1:
        buff_size += 1
        
        
    err_h_all = []
                                    
    err_l_all = []
    
    dictall_high = []
    dictall_low = []
    #####################################3
    
    #entering optimization....
    for k in range(train_iter):
        
        ########################################################
        #cluster calculation start!    
        ttime3 = time.time()
         

        ttime2 = time.time()
        
        ###dictionaries calculations and broadcasting to the cluster.
        dict_ht_bc = sc.broadcast(np.squeeze(np.transpose(dict_h)))
        dict_lt_bc = sc.broadcast(np.squeeze(np.transpose(dict_l)))
    	
        
        dtdh = np.dot(np.squeeze(np.transpose(dict_h)), np.squeeze(dict_h)) +  (c1 + c3)*np.eye(np.squeeze(np.transpose(dict_h).shape[0]))
        dtdhinv_bc = sc.broadcast(inv(dtdh))
        
        dtdl = np.dot(np.squeeze(np.transpose(dict_l)), np.squeeze(dict_l)) +  (c2 + c3)*np.eye(np.squeeze(np.transpose(dict_l).shape[0]))
        dtdlinv_bc = sc.broadcast(inv(dtdl))
        
        time_upd = time.time()
        
        #update the CDL object    
        tmp = tmp.map(lambda x: updateCD(x,dict_ht_bc.value, dict_lt_bc.value, dtdhinv_bc.value, dtdlinv_bc.value,c1,c2,c3,k)).cache()
        
        
        print('time elapsed for updating...')
        print(time.time() - time_upd)
      
        
        time_calc = time.time()
        
        
        #extract the SxW and phi matrices for low and high resolution.
        updvals = tmp.map(lambda x: getVals(x)).reduce(lambda x,y: calcSum3(x, y))
        
        print('time elapsed for calculating...')
        print(time.time() - time_calc)
        
        print('****************************************')
        
        
        print('time elapsed:')
        print time.time()- ttime2
        #cluster calculation done!
        ##########################################################
        
        sw_h =np.array(updvals[0][0])
        sw_l =np.array(updvals[0][1]) 
        print(sw_h.shape)
        phi_h = np.array(updvals[1][0])
        phi_l = np.array(updvals[1][1])
        print(phi_h.shape)
        
        phi_h = np.reshape(phi_h, ((1, len(phi_h)))) + delta
        
        #calculate and normalize the new dictionaries
        #a. high resolution
        dict_h_upd = dict_h + sw_h/(phi_h)
        dict_h = normD(dict_h_upd)
        #b. low resolution
        phi_l = np.reshape(phi_l, ((1, len(phi_l)))) + delta
        dict_l_upd = dict_l + sw_l/(phi_l)
        dict_l = normD(dict_l_upd)
        
        #clean up your garbage!
        sc._jvm.System.gc()

        #######################################################
        
        #error calculation over the cluster
        if (k + 1) % wind == 0:
            
            err_all = tmp.map(lambda x: calcErr(dict_h, dict_l,x)).reduce(lambda x,y: calcSum2(x,y))
        
            err_h = math.sqrt(np.sum(np.array(err_all[0])) / (bands_h_N * imageN))
            err_l = math.sqrt(np.sum(np.array(err_all[1])) / (bands_l_N * imageN))
            
            
            print('ERROR HIGH:')
            print(err_h)
            
            print('ERROR LOW:')
            print(err_l)
            sc._jvm.System.gc()
            
            ##############################
            
            #append errors & dictionaries (for checking for convergence over a sliding window)
            err_h_all.append(err_h)
            err_l_all.append(err_l)
            
            dictall_high.append(dict_h)
                
            dictall_low.append(dict_l)
            
            
            ##check error convergence >>>start!
            if m >= buff_size - 1:
                print('checking for error convergence...')
                tmp_h = err_h_all[m-buff_size + 1:m+1]
                tmp_l = err_l_all[m-buff_size + 1:m+1]
                
                err_con_h = np.mean(tmp_h[:buff_size/2], axis=0) - np.mean(tmp_h[buff_size/2:], axis=0)
                err_con_l = np.mean(tmp_l[:buff_size/2], axis=0) - np.mean(tmp_l[buff_size/2:], axis=0)
                
                if (abs(err_con_h) > err_thrs or abs(err_con_l) > err_thrs) and k >= train_iter/2 + 1 :
                    
                    minindex_l = np.array(tmp_h).argmin()
                
                    minindex_h = np.array(tmp_l).argmin()
                
                
                    dict_h = np.array(dictall_high[minindex_h])
                    
                    dict_l = np.array(dictall_low[minindex_l])
                    print('break!')
                    
                    break
            
            
                if mm >= buff_size:
                    dictall_high = []
                    dictall_low = []
                    mm = 0
                print('...done!')    
                
            m = m + 1
            mm = mm + 1
        ########################################
        
        #when done with this iteration remove the dictionaries from the cluster (to improve memory efficiency)
            
        dict_ht_bc.unpersist(blocking=True)
        
 
        dict_lt_bc.unpersist(blocking=True)
           

        
        dtdhinv_bc.unpersist(blocking=True)
        
        dtdlinv_bc.unpersist(blocking=True)
        
        ########################################
        
        print('Time elapsed for this iteration: ')
        ttime3 = time.time()-ttime3
        print(ttime3)
        time_all.append(ttime3)
        #print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        
        ##optional: save the results of each iteration in a mat file.
        #mat2save =  './results' + str(imageN) + 'x' + str(k)+  '_'  + str(dictsize) + '_' +str(partitions_N) + '.mat'  
        
        #sio.savemat(mat2save, {'timeelapsed': ttime3, 'dicth':dict_h, 'dictl': dict_l, 'phi_h': phi_h, 'phi_l': phi_l, 'sw_h': sw_h, 'sw_l': sw_l, 'err_l':err_l, 'err_h': err_h})#, 'wh': wh, 'wl': wl})#'phih': phi_h, 'sw': sw})
        
        
    #save error values and final dictionaries in a mat file.
        
    mat2savefin =  './results_fin' + str(imageN) + 'x' + str(dictsize) + '_' +str(partitions_N) + '.mat'  
        
    sio.savemat(mat2save, {'dicth':dict_h, 'dictl': dict_l, 'err_l': err_h_all, 'err_h': err_l_all, 'time_all': time_all})
        
    return 1

        
def main(args=None):

    sc = SparkContext()
    sqlContext = SQLContext(sc)
    global opts
    opts = get_opts(args)
    run_script(sc)
    sc.stop()


if __name__ == "__main__":
    main()

