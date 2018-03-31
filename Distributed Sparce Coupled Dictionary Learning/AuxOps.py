"""Distributed Sparse Coupled Dictionary Learning

Sumplementary Class and Methods for the calculation of intermediate matrices (Lagrange multiplier matrices, sparse coding matrices)

:Author: Nancy Panousopoulou <apanouso@ics.forth.gr>

:[1] K.  Fotiadou, G. Tsagkatakis, P. Tsakalides `` Linear Inverse Problems with Sparsity Constraints,'' DEDALE DELIVERABLE 3.1, 2016.
 
:[2] K. Fotiadou, G. Tsagkatakis, P. Tsakalides. Spectral Resolution Enhancement via Coupled Dictionary Learning. 2017. Under Review in IEEE Transactions on Remote Sensing.

:Date: 01/02/2018
"""



import numpy as np

from numpy.linalg import inv


from scipy.linalg import solve 

    
def updateY(previousY, c, op1, op2, maxbeta=1e+6, beta=0.01):
    """ Update Lagrange multiplier matrix (Equation (19)). 
    
    Input Arguments
    ----------
    previousY : np.array (RDD block)
        The previous value of the Lagrange multiplier matrix
    c: double
        Step size parameter for the augmentend Lagrangian function
    op1: np.array
        First operand matrix (thresholding values)
    op2 : np.array
        Second operand matrix (sparse coding coefficients)
    maxbeta, beta: double
        Auxiliary parameters for updating Lagrange multiplier matrices.
    
    
    """     
    return  previousY + min(maxbeta, beta*c)*(op1-op2)
    #return previousY+np.ones((previousY.shape))
  
    
def updThr(inputmat, lam=0.1):
    
    """
    Update thresholding matrices (Equations (13)-(14)). 
    
    Input Arguments
    ----------
    inputmat : np.array (RDD block) 
        The input matrix for thresholding
    lam: double
        The thresholding value
        
    
    """
    
        
    th = lam/2.
    k= 0
    ttt = np.random.random(inputmat.shape)
   
    
    for aa in inputmat:
       
        if aa>th:
            ttt[k] = aa-th
        elif abs(aa) <= th:
            ttt[k] = 0.
        elif aa < (-1.)*th:
            ttt[k] = aa +th
            
            
        k +=1    
            
            
    return ttt


def calcW(datain_h, datain_l, dictin_ht, dictin_lt, dtdh, dtdl, wh,wl, c1,c2,c3, y1,y2,y3,p,q, cnt, dictsize):
    """
    Update sparse coding parameters (Equation (11)) over the cluster. 
    
    Input Arguments
    ----------
    datain_h, datain_l : np.array  (RDD block)
        The input data matrices in high and low resolution respectively
    dictin_ht, dictin_lt: np.array 
        The transpose of the input dictionary in high and low resolution respectively
    wh, wl: np.array (RDD block)
        The previous sparse coding matrices in high and low resolution respectively
    c1,c2,c3: double
        Step size parameters for the augmented Lagrangian function
    y1,y2,y3: np.array
        The Langrange multiplier matrices
    p,q: np. array
        The thresholding matrices for high and low resolution respectively.
        
        
    Returns:
    -------------
    the updated RDD blocks for sparse coding coefficients
    
    """
            
    
    
           
    tmp2 = np.dot(dictin_ht, np.transpose(np.array(datain_h))) + (y1 - y3) + c1*p + c3*wl
    
    tmp22 = np.dot(dtdh, tmp2)
        
    tmp4 = np.dot(dictin_lt, np.transpose(np.array(datain_l))) + (y2 - y3) + c2*q + c3*wh
    
    tmp44 = np.dot(dtdl, tmp4)
    return [tmp22, tmp44]
    
    
