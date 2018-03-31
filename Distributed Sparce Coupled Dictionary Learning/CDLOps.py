"""Distributed Sparse Coupled Dictionary Learning

Sumplementary Class and Methods for the calculation of intermediate matrices (Lagrange multiplier matrices, sparse coding matrices)

:Author: Nancy Panousopoulou <apanouso@ics.forth.gr>

:[1] K.  Fotiadou, G. Tsagkatakis, P. Tsakalides `` Linear Inverse Problems with Sparsity Constraints,'' DEDALE DELIVERABLE 3.1, 2016.
 
:[2] K. Fotiadou, G. Tsagkatakis, P. Tsakalides. Spectral Resolution Enhancement via Coupled Dictionary Learning. 2017. Under Review in IEEE Transactions on Remote Sensing.

:Date: 01/02/2018
"""

import numpy as np
import AuxOps as aops
import time
import copy

class CDL():
    """Coupled Dictionary Learning Class

    This class defines the intermediate matrices ((Lagrange multiplier matrices, sparse coding matrices)

    Parameters
    ----------
    datain : tuple of (np.ndarray, np.ndarray)
        Input RDD block of the buddled low and high resolution data 
        
    dictsize : int
        The size of the dictionaries
    """        
    
    def __init__(self, datain, dictsize):
       
        self.datain_h = np.array(datain[0])
        self.datain_l = np.array(datain[1])
        #the sparse coefficients
        self.wh = np.zeros((dictsize,1))
        self.wl = np.zeros((dictsize,1))
        
        self.p = np.zeros((dictsize, 1))
        self.q = np.zeros((dictsize, 1))
        #the Lagrange multiplier matrices 
        self.y1 = np.zeros((dictsize, 1))
        self.y2 = np.zeros((dictsize, 1))
        self.y3 = np.zeros((dictsize, 1))
        
        self.dictsize = dictsize
        

    
def updateCD(cdlin, dictin_ht, dictin_lt, dtdh, dtdl, c1,c2,c3,cnt):
    
    """
    Method for updating intermediate matrices over the cluster 

    Input Arguments
    ----------
    cdlin : CDL object (RDD block)
        The set of intermediate matrices to be updated
    dictin_ht: np.array
        The transpose of the input dictionary in high resolution
    dictin_lt: np.array
        The transpose of the input dictionary in low resolution    
    dtdh: np.array
        Auxiliary matrix - first term of Equation (11) for the high resolution dictionaries
    dtdl: np.array
        Auxiliary matrix - first term of Equation (11) for the low resolution dictionaries
    c1, c2, c3: double
        Step size parameters for the augmentend Lagrangian function
    maxbeta, beta: double
        Auxiliary parameters for updating Lagrange multiplier matrices.
    lamda: double
        The threshold value.
        
        
    Returns:
    ----------------
    The updated CDL object (matrices W, P,Q, Y1,Y2,Y3)
    
    """    
    
    
    y11= np.squeeze(cdlin.y1)
    y22 = np.squeeze(cdlin.y2)
    
    y33 = np.squeeze(cdlin.y3)
    pp = np.squeeze(cdlin.p)
    
    qq = np.squeeze(cdlin.q)
    
        
                     
    datain_h = np.array(cdlin.datain_h)
    datain_l = np.array(cdlin.datain_l)
        
    wl = np.squeeze(cdlin.wl)
    wh = np.squeeze(cdlin.wh)
    #update the sparse coding matrices according to Eq. (11)    
    whl = aops.calcW(datain_h, datain_l, np.squeeze(dictin_ht),np.squeeze(dictin_lt), dtdh, dtdl, wh, wl, c1,c2,c3, y11,y22,y33,pp,qq, cnt,cdlin.dictsize)
    #update the thresholding matrices according to Eq. (13)  
    pp = aops.updThr(whl[0]-y11/c1)

    qq = aops.updThr(whl[1]-y22/c2)
    #update the Lagrange multiplier matrices according to Eq. (19).
    y11 = aops.updateY(y11, c1,  pp, whl[0])
    y22 = aops.updateY(y22, c1,  qq, whl[1])
    y33 = aops.updateY(y33, c3,  whl[0], whl[1])   
    
        
    cdlin.wh = whl[0].copy()
    cdlin.wl = whl[1].copy()
    
    cdlin.y1 = y11.copy() 
    cdlin.y2 = y22.copy()
    cdlin.y3 = y33.copy()
    cdlin.p = pp.copy()
    cdlin.q = qq.copy()
        
    return cdlin
            

def startCDL(datain, dictsize):
    """Auxiliary method for instatiating a CDL object over the cluster.

    
    Parameters
    ----------
    datain : tuple of (np.ndarray, np.ndarray)
        Input RDD block of the buddled low and high resolution data 
        
    dictsize : int
        The size of the dictionaries
        
    Returns:
    ----------
    the cdl object (RDD blocks)
    
    """
    
    
    mycdl=CDL(datain, dictsize)
    
    return mycdl

    
    
def calcOutProd(ina, inb):
    """Outer product calculation between two nparrays (auxiliary for distributed matrix product)
    
    Parameters
    ----------
    ina, inab : nparray (1D)
                Input arrays
    Returns:
    ----------
    The outer product of ina, inb 
    
    """
    
    return np.dot(ina, inb)
