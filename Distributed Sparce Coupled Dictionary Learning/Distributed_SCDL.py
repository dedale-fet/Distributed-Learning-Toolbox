
import numpy as np
import scipy.io
import os.path
from pyspark import SparkContext
from pyspark.sql import * 
#from pyspark.sql.types import *
import pyspark.sql.types as pst

#from pyspark.ml.feature import Normalizer
#from pyspark.ml.linalg import Vectors
from subprocess import call
import time

from CDLOps import *
from numpy import linalg as la

import math
from numpy import genfromtxt

from functools import partial

import scipy.io as sio
from numpy.linalg import inv
import copy

from args import get_opts


def calcSum2(data1, data2):
    
    #for i in range(0, len(data1)):
    data11 = data1[0]
    data12 = data1[1]
    
    data21 = data2[0]
    data22 = data2[1]
    
    
    ph1 = [data11[i] + data21[i] for i in range(0,len(data11))]
    ph2 = [data12[i] + data22[i] for i in range(0,len(data12))]
    
    #ph = data1 + data2
    
    return [ph1, ph2]



def calcSum3(data1, data2):
    
    #for i in range(0, len(data1)):
    data111 = data1[0][0]
    data112 = data1[0][1]
    data121 = data1[1][0]
    data122 = data1[1][1]
    
    data211 = data2[0][0]
    data212 = data2[0][1]
    data221 = data2[1][0]
    data222 = data2[1][1]
    
    
    #ph1 = [data11[i] + data21[i] for i in range(0,len(data11))]
    #ph2 = [data12[i] + data22[i] for i in range(0,len(data12))]
    
    #(sw_h, sw_l), (phi_h, phi_l)
    
    swh = [data111[i] + data211[i] for i in range(0,len(data111))]
    swl = [data112[i] + data212[i] for i in range(0,len(data112))]
    phih = [data121[i] + data221[i] for i in range(0,len(data121))]
    phil = [data122[i] + data222[i] for i in range(0,len(data122))]
    
    
  
    
    #ph = data1 + data2
    
    return [[swh, swl], [phih, phil]]

def calcErrI(dicth,dictl,bands_h_N, bands_l_N,x):
    
    
    
    errh = []
    errl = []
    
    for ii in iter(x):
        print('getting errors...')
        print(len(ii))
        aa = ii
        eh = np.zeros((bands_h_N,1))
        el = np.zeros((bands_l_N,1))
        
        for cdli in aa:
            
            eh = np.sum((np.square((cdli.datain_h - np.dot(dicth, np.transpose(cdli.wh)))), eh))
            el = np.sum((np.square((cdli.datain_l - np.dot(dictl, np.transpose(cdli.wl)))), el))
            #tmp2 = np.square((cdli.datain_l - np.dot(dictl, np.transpose(cdli.wl))))
    
        errh.append(el)
        errl.append(eh)
         
    yield [errh, errl]
    
    
def calcErr(dicth,dictl,cdli):
    
             
    errh = np.square((cdli.datain_h - np.dot(dicth, np.transpose(cdli.wh))))
    errl = np.square((cdli.datain_l - np.dot(dictl, np.transpose(cdli.wl))))
    
         
    return [errh, errl]


def getVals1(cdli):
    
    #tv = time.time()
    
    swh = calcOutProd(np.reshape(cdli.datain_h, (cdli.datain_h.shape[0],1)), np.reshape(cdli.wh, (1, cdli.wh.shape[0])))
    swl = calcOutProd(np.reshape(cdli.datain_l, (cdli.datain_l.shape[0],1)), np.reshape(cdli.wl, (1, cdli.wl.shape[0])))        
    phi_h = [cdli.wh[i]*cdli.wh[i] for i in range(0,len(cdli.wh))] #np.square(cdli.wh)
    phi_l = [cdli.wl[i]*cdli.wl[i] for i in range(0,len(cdli.wl))]
    
    #print('time in get values:')
    #print(time.time() - tv)   
    #print('**********************')
    #print(np.array(wh).shape)     
    return [[swh, swl], [phi_h, phi_l]]    
     
   
def normD(dictin):
    
    tmp = 1 / np.sqrt(np.sum(np.multiply(dictin, dictin), axis=0))
    return np.dot(dictin, np.diag(tmp))



#def main(partitions_N=1):
def run_script(sc):

    # some explicit variables and parameter
    imageN = opts.imageN #100#39998#100#79996 #39998#79996#39998 # max=79996
    dictsize = opts.dictsize#64#512#64#512#256#64#1024#256                                                                                                                                                                          # 1024 #512 #1024 #512 #1024  # 1024 when image_N = 79996
    bands_h_N = opts.bands_h
    bands_l_N = opts.bands_l
    c1 = 0.4
    c2 = 0.4
    c3 = 0.8
    maxbeta = 1e+6#pow(10,6)
    delta = 1e-4#*np.ones((dictsize,1))
    beta = 0.01
    lamda = opts.lamba
    train_iter =opts.n_iter
    
    partitions_N = opts.partitions
    
    #data_h = sc.parallelize(genfromtxt('/home/sparkuser/nancy/cdl/hs/512x39998/Xh.csv', delimiter=','), partitions_N).cache()
    #data_l = sc.parallelize(genfromtxt('/home/sparkuser/nancy/cdl/hs/512x39998/Xl.csv', delimiter=','), partitions_N).cache()
    
    data_h = sc.parallelize(genfromtxt(opts.inputhigh, delimiter=','), partitions_N).cache()
    data_l = sc.parallelize(genfromtxt(opts.inputlow, delimiter=','), partitions_N).cache()

     
    
    dict_h_t = data_h.take(dictsize)#.collect()
    dict_l_t = data_l.take(dictsize)#sample(False, 0.013, 0).collect()
   
    
    dict_h = np.transpose(dict_h_t)
    
    #dict_h = dict_h / la.norm(dict_h)
    
    dict_l = np.transpose(dict_l_t)
    #dict_l = dict_l /la.norm(dict_l)
    print('>>>>>>>>>>>>>>>>')
    print(data_h.count())
    print(data_l.count())
    datain = data_h.zip(data_l).cache()
    print('>>>>>>>>>>>>>>>>')
   
    print(dict_h.shape)
    print(dict_l.shape)
 
    mat2save =  './tttmpinit' + '_' + str(imageN) + 'x' + str(dictsize) + '.mat'  
    sio.savemat(mat2save, {'dicth_init':dict_h, 'dictl_init': dict_l})#'phih': phi_h, 'sw': sw})
   
    lamda_bc = sc.broadcast(lamda)
    dictsize_bc = sc.broadcast(dictsize)
    
    dict_h_bc = sc.broadcast(dict_h)
    dict_l_bc = sc.broadcast(dict_l)
    
    
    tmp = datain.map(lambda x: startCDL(x, dictsize_bc.value)).cache() 
    
    
  
    wind = opts.window;    
    #print(tmp.y1.shape)
    err_h = 0;
    err_l = 0;
    
    buff_size = 10 #to do - put this in arg in...
    err_thrs = 0.01 #likewise
    
    m = 0
    mm = 0
    
    ##keep the error buffer size as an even number.
    if buff_size % 2 == 1:
        buff_size += 1
        
        
    err_h_all = []
                                    
    err_l_all = []
    
    dictall_high = []
    dictall_low = []
    
    
  
    for k in range(train_iter):    
        ttime3 = time.time()
         

        ttime2 = time.time()
        
        
        dict_ht_bc = sc.broadcast(np.squeeze(np.transpose(dict_h)))
        dict_lt_bc = sc.broadcast(np.squeeze(np.transpose(dict_l)))
    	#dict_ht = np.squeeze(np.transpose(dict_h))
        #dict_lt = np.squeeze(np.transpose(dict_l))
   
        
        dtdh = np.dot(np.squeeze(np.transpose(dict_h)), np.squeeze(dict_h)) +  (c1 + c3)*np.eye(np.squeeze(np.transpose(dict_h).shape[0]))
        dtdhinv_bc = sc.broadcast(inv(dtdh))
        #dtdhinv = inv(dtdh)
        dtdl = np.dot(np.squeeze(np.transpose(dict_l)), np.squeeze(dict_l)) +  (c2 + c3)*np.eye(np.squeeze(np.transpose(dict_l).shape[0]))
        dtdlinv_bc = sc.broadcast(inv(dtdl))
        #dtdlinv = inv(dtdl)
        
        time_upd = time.time()    
        tmp = tmp.map(lambda x: updateCD(x,dict_ht_bc.value, dict_lt_bc.value, dtdhinv_bc.value, dtdlinv_bc.value,c1,c2,c3,k)).cache()
        #tmp = tmp.map(lambda x: updateCD(x,dict_ht, dict_lt, dtdhinv, dtdlinv,c1,c2,c3,k)).cache()
     
        
        print('time elapsed for updating...')
        print(time.time() - time_upd)
      
        
        time_calc = time.time()
        
        '''
        ttmp1= np.array(tmp1.collect())
        wht =np.array(ttmp1[0])
        wlt = np.array(ttmp1[1])
        
        print(wht)
        print('******************8')
        print(wlt)
        '''
        updvals = tmp.map(lambda x: getVals1(x)).reduce(lambda x,y: calcSum3(x, y))
        
        
        
        
        print('time elapsed for calculating...')
        print(time.time() - time_calc)
        
        print('****************************************')
        #hghfd
        #tmp.count()
        
        print('time elapsed:')
        print time.time()- ttime2
        
        
        
        sw_h =np.array(updvals[0][0])
        sw_l =np.array(updvals[0][1]) 
        print(sw_h.shape)
        phi_h = np.array(updvals[1][0])
        phi_l = np.array(updvals[1][1])
        print(phi_h.shape)
        #phi_h = phiupd[0]
        #phi_l = phiupd[1]
        
        #sw_h =swupd[0] 
        #sw_l =swupd[1]
    #collect()#lambda x: [x.swh, x.swl])
        #print(sw_h.shape)
        
        #phi_h = phi_h# + delta*np.ones((len(phi_h)))
        #print(phi_h.shape)
        
        phi_h = np.reshape(phi_h, ((1, len(phi_h)))) + delta
        dict_h_upd = dict_h + sw_h/(phi_h)
        dict_h = normD(dict_h_upd) #dict_h_upd/la.norm(dict_h_upd)
        
        phi_l = np.reshape(phi_l, ((1, len(phi_l)))) + delta
        dict_l_upd = dict_l + sw_l/(phi_l)
        dict_l = normD(dict_l_upd) # dict_l_upd/la.norm(dict_l_upd)
        
        sc._jvm.System.gc()

        #dict_ht_bc.unpersist()
        #dict_ht_bc.destroy()
 
        #dict_lt_bc.unpersist()
        #dict_lt_bc.destroy()

	#dict_ht = np.squeeze(np.transpose(dict_h))
        #dict_lt = np.squeeze(np.transpose(dict_l))
   

        
        #dtdhinv_bc.unpersist()
        #dtdhinv_bc.destroy()

        #dtdlinv_bc.unpersist()
        #dtdlinv_bc.destroy()

        ##calculate error every 10 iterations...
        
    
        
        #if (k +1) % wind:
        #    err_h = err_h
        #    err_l = err_l
            #continue
        
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
            err_h_all.append(err_h)
            err_l_all.append(err_l)
            
            dictall_high.append(dict_h)
                
            dictall_low.append(dict_l)
            
            
            
            if m >= buff_size - 1:
                print('checking for error convergence...')
                #print(m)
                #print(m-buff_size+1)
                tmp_h = err_h_all[m-buff_size + 1:m+1]
                tmp_l = err_l_all[m-buff_size + 1:m+1]
                
                err_con_h = np.mean(tmp_h[:buff_size/2], axis=0) - np.mean(tmp_h[buff_size/2:], axis=0)
                err_con_l = np.mean(tmp_l[:buff_size/2], axis=0) - np.mean(tmp_l[buff_size/2:], axis=0)
                
                #print(tmp_h[:buff_size/2])
                #print(tmp_h[buff_size/2:])
                
                #print(tmp_h)
                #print(tmp_l)
                
                
                if (abs(err_con_h) > err_thrs or abs(err_con_l) > err_thrs) and k >= train_iter/2 + 1 :
                    
                    minindex_l = np.array(tmp_h).argmin()
                
                    minindex_h = np.array(tmp_l).argmin()
                
                    #print(minindex_h)
                    #print(len(dictall_high))
                
                    dict_h = np.array(dictall_high[minindex_h])
                    
                    dict_l = np.array(dictall_low[minindex_l])
                    print('break!')
                    #print(dict_h)
                    #print(dict_l)
                    
                    break
            
            
                if mm >= buff_size:
                    dictall_high = []
                    dictall_low = []
                    mm = 0
                print('...done!')    
                
            m = m + 1
            mm = mm + 1
        ########################################    
        dict_ht_bc.unpersist(blocking=True)
        #dict_ht_bc.destroy()
 
        dict_lt_bc.unpersist(blocking=True)
        #dict_lt_bc.destroy()

        #dict_ht = np.squeeze(np.transpose(dict_h))
        #dict_lt = np.squeeze(np.transpose(dict_l))
   

        
        dtdhinv_bc.unpersist(blocking=True)
        #dtdhinv_bc.destroy()

        dtdlinv_bc.unpersist(blocking=True)
        #dtdlinv_bc.destroy()            
        #print(err_h)
        
        #print(err_l)
        
        
        #tmp.unpersist()
        
        print('Time elapsed for this iteration: ')
        ttime3 = time.time()-ttime3
        print(ttime3)
        #print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        mat2save =  './results' + str(imageN) + 'x' + str(k)+  '_'  + str(dictsize) + '_' +str(partitions_N) + '.mat'  
        
        sio.savemat(mat2save, {'timeelapsed': ttime3, 'dicth':dict_h, 'dictl': dict_l, 'phi_h': phi_h, 'phi_l': phi_l, 'sw_h': sw_h, 'sw_l': sw_l, 'err_l':err_l, 'err_h': err_h})#, 'wh': wh, 'wl': wl})#'phih': phi_h, 'sw': sw})
        #sio.savemat(mat2save, {'timeelapsed': ttime3, 'dicth':dict_h, 'dictl': dict_l, 'err_h': err_h, 'err_l': err_l})#'phih': phi_h, 'sw': sw})
    
        
        



def main(args=None):

    sc = SparkContext()
    sqlContext = SQLContext(sc)
    global opts
    opts = get_opts(args)
    run_script(sc)
    sc.stop()


if __name__ == "__main__":
    main()

