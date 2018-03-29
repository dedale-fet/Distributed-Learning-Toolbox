##spark auxiliary functions
"""Distributed Learning Module for PSF DECONVOLUTION (cluster module)

This module contains all auxiliary methods for performing the PSF deconvolution over the distributed learning architecture

:Author: Nancy Panousopoulou@FORTH-ICS (apanous@ics.forth.gr)


:Release: 13/12/2017 


"""





import numpy as np
from sf_tools.signal.wavelet import filter_convolve_stack, filter_convolve
from sf_tools.signal.linear import *
from sf_tools.signal.optimisation import *
from sf_tools.image.convolve import psf_convolve
from sf_tools.signal.noise import thresh
from sf_tools.signal.svd import svd_thresh
from sf_tools.base.transform import *

from scipy.linalg import norm
from sf_tools.math.matrix import nuclear_norm





########################################################################################
def mycalc_norm(data,t):
    """
    Auxiliary method for norm calculation over a block of RDD data (cluster) considering a constant threshold
    
    Input arguments
    ----------------
    data: np.ndarray(1x3D)
          input data (RDD blocks)
            
    t: np.ndarray (1x3D) 
        threshold for norm calculation
    
    
    Returns
    -----------------
    the norm value for the RDD block
    
    
    """
    return np.array([[norm(b) * c * np.ones((data.shape[2], data.shape[3])) for b, c in zip(a, t)] for a in data])
    
########################################################################################

########################################################################################
def sp_psf_convolve(combined_data, rot=True, data_type='fixed'):
    
    
    """
    Auxiliary method for performing convolution over a bundled block of RDD data (cluster)
    
    Input arguments
    ----------------
    combined_data: tuple in the form (np.ndarray(3D), np.ndarray(3D))
                   input tuple of data (RDD blocks)
                   
    rot: boolean 
        rotation parameter for convolution
    
    data_type: str{'fixed', 'obj_var'}
                The PSF type (fixed or spatially variant)
    
    
    Returns
    -----------------
    the convolution over the contents of the tupled RDD block
    
    
    """
    
    
    
	#split and convert to arrays
    data1, data2 = zip(*combined_data)
    
    np_data1 = np.array(data1)
    np_data2 = np.array(data2)
    
    #calculate the convolution
    
    return np.reshape(np.array(psf_convolve(np_data1, np_data2, psf_rot=rot, psf_type=data_type)), (np_data1.shape[1],np_data2.shape[2]))#spec_rad #in 1#spec_rad
    
##################################################################################



#####################################################################################
def sp_runupdate_sparse(din,linear_filters, rho,sigma,tau,wv_thr_factor, noise_est):

    """
    Sparsity-based update of the primal-dual optimization variables (Condat optimization) over a bundled block of RDD data (cluster)
    
    Input arguments
    ----------------
    din: data bundle in the form (np.ndarray(3D), np.ndarray(3D), np.ndarray(3D), np.ndarray(3D))
                   input tuple of data (RDD blocks)
                   
    linear_filters: np.array
                    The wavelet filters
    
    pho, sigma, tau: float
                     optimization parameters
                     
    wv_thr_factor:  np.arrray (1x3)
                    the wavelet threshold factor
                    
                    
     noise_est: float
               The noise standard deviation in the observed galaxy images.
    
    
    Returns
    -----------------
    the updated bundled block of RDD data
    
    
    """
    
    
    
    
    #unbuddle the RDD block into (datain, psfin,xin, yin)
     
    tmp, yin = zip(*zip(*zip(*zip(*zip(din)))))
    
    tmp11, xin = zip(*zip(*zip(*tmp)))
    datain, psfin = zip(*zip(*zip(*tmp11)))
    
    tauin = tau
    sigmain = sigma
    rhoin = rho
    extrafactor = 1.0 / sigmain
    
    psfinrot = np.rot90(psfin,2)
    win = noise_est* mycalc_norm(filter_convolve_stack(psfinrot,linear_filters), wv_thr_factor)
    
    gtemp1 = psf_convolve(xin,psfin,psf_rot=False, psf_type='obj_var')
    
    gtemp = psf_convolve(gtemp1 - datain, psfin, psf_rot=True, 
                         psf_type='obj_var')
   
                         
    xtemp = xin - tauin * gtemp - tauin * filter_convolve_stack(yin, linear_filters, filter_rot=True)
    
   
    
    
    x_prox = xtemp * (xtemp > 0)
    
    # Step 2 from eq.9.
    threshold = np.squeeze(np.array(win)) * extrafactor
        
    
    ytemp = yin + sigmain*filter_convolve_stack(2*x_prox-xin, linear_filters)
    yytemp = thresh(ytemp / sigmain,threshold, threshold_type='soft')
    
    y_prox = (ytemp - sigmain * yytemp)
    
    
    # Step 3 from eq.9.
    xout = rhoin * x_prox + (1 - rhoin) * np.array(xin)
    yout = rhoin * y_prox + (1 - rhoin) * np.array(yin)
    
    
    #combine and return the buddled RDD block with the updated primal (xout) and dual (yout) parameters
    tt = zip(zip(zip(datain,psfin),xout),yout)
    
    tt = tuple(tt)
    return tt[0]
    
    
############################################################################





########################################################################
def sp_runupdate_lowr(din, lamb, rho, sigma, tau, thres_type = 'hard'):
    """
    Low rank-based update of the primal-dual optimization variables (Condat optimization) over a bundled block of RDD data (cluster)
    
    Input arguments
    ----------------
    din: data bundle in the form (np.ndarray(3D), np.ndarray(3D), np.ndarray(3D), np.ndarray(3D))
                   input tuple of data (RDD blocks)
                   
    lamb: float
          the low-rank threshold  
          
    
    pho, sigma, tau: float
                     optimization parameters
                     
    thres_type: str{'hard', 'soft'}
                Type of noise to be added (default is 'hard')
    
    
    Returns
    -----------------
    the updated bundled block of RDD data
    
    
    """


    #unbuddle the RDD block into (datain, psfin,xin, yin)
    
    tmp, yin = zip(*zip(*zip(*zip(*zip(din)))))
    tmp11, xin = zip(*zip(*zip(*tmp)))
    datain, psfin = zip(*zip(*zip(*tmp11)))
    
    
    
    tauin = tau
    sigmain = sigma
    rhoin = rho
    extrafactor = 1.0 / sigmain
    
    xin = np.array(xin)
    
    yin = np.array(yin)
    
    gtemp1 = psf_convolve(xin,psfin,psf_rot=False, psf_type='obj_var')
    
    gtemp = psf_convolve(gtemp1 - datain, psfin, psf_rot=True, 
                         psf_type='obj_var')
                         
    xtemp = xin - tauin * gtemp - tauin * yin
    
   
    
    
    x_prox = xtemp * (xtemp > 0)
    
    threshold = lamb * extrafactor
        
    
    ytemp = yin + sigmain*(2*x_prox-xin)
    
    yytemp = svd_thresh(cube2matrix(ytemp/sigmain), threshold, n_pc='all', thresh_type=thres_type)

    
    yytemp = matrix2cube(yytemp, ytemp.shape[1:])

    
    
    y_prox = (ytemp - sigmain * yytemp)
    
    
    # Step 3 from eq.9.
    xout = rhoin * x_prox + (1 - rhoin) * np.array(xin)
    yout = rhoin * y_prox + (1 - rhoin) * np.array(yin)
    
    
    #combine and return the buddled RDD block with the updated primal (xout) and dual (yout) parameters
    tt = zip(zip(zip(datain,psfin),xout),yout)

    tt = tuple(tt)
    return tt[0]
    
    
################################################################################

########################################################################
## check convergence
########################################################################
def sp_calc_cost_sparse(din, linear_filters,wv_thr_factor, noise_est):
    
    """
    This method calculates the distributed value of the cost function for the sparse-based Condat optimization
    
     
    Input arguments
    ----------------
    din: data bundle in the form (np.ndarray(3D), np.ndarray(3D), np.ndarray(3D), np.ndarray(3D))
                   input tuple of data (RDD blocks)
                   
    linear_filters: np.array
                    The wavelet filters
    
                         
    wv_thr_factor:  np.arrray (1x3)
                    the wavelet threshold factor
    
    noise_est: float
               The noise standard deviation in the observed galaxy images.
    
    Returns
    -----------------
    the updated bundled block of RDD data
    
    
    Returns
    -----------------
    the cost value over the RDD blocks
    
    
    """
    
    
    tmp, yin = zip(*zip(*zip(*zip(*zip(din)))))
    tmp11, xin = zip(*zip(*zip(*tmp)))
    datain, psfin = zip(*zip(*zip(*tmp11)))
    
    
    
    
    psfinrot = np.rot90(psfin,2)
    win = noise_est* mycalc_norm(filter_convolve_stack(psfinrot,linear_filters), wv_thr_factor)
   
    #term 1: data - filter
    gtemp1 = psf_convolve(xin,psfin,psf_rot=False, psf_type='obj_var')
    
    
    d1 = datain - gtemp1
    #term 2: 
    d2 = np.squeeze(np.array(win)) * filter_convolve_stack(xin, linear_filters)
   
    cost = 0.5*np.linalg.norm(d1)**2 + np.sum(np.abs(d2))
   
    return cost
#################################################################################

#################################################################################
def sp_calc_cost_lowr(din, lamb):
    """

    This method calculates the distributed value of the cost function for the low rank-based Condat optimization
    
     
    Input arguments
    ----------------
    din: data bundle in the form (np.ndarray(3D), np.ndarray(3D), np.ndarray(3D), np.ndarray(3D))
                   input tuple of data (RDD blocks)
                   
    lamb: float
          the low-rank threshold  
    
    
    Returns
    -----------------
    the cost value over the RDD blocks
    

   
    
    """
    
    
    
    tmp, yin = zip(*zip(*zip(*zip(*zip(din)))))
    
    tmp11, xin = zip(*zip(*zip(*tmp)))
    datain, psfin = zip(*zip(*zip(*tmp11)))
    
    xin = np.array(xin)
    
    yin = np.array(yin)

    #term 1: data - filter

    gtemp1 = psf_convolve(xin,psfin,psf_rot=False, psf_type='obj_var')
    
    
    
    d1 = datain - gtemp1
    
   
    cost = 0.5*np.linalg.norm(d1)**2
   
    return cost
###################################################################################


###################################################################################
def sp_check_convergence(cost_list, window, tolerance):
    """
    Auxiliary method for checking the convergence of the cost function
    (redefined for convenience from sf_tools.signal.cost)
    
    Input arguments
    ----------------
    cost_list: list
                   
    window: integer 
            sliding window length
    
    tolerance: float
               The threshold for checking the convergence
    
    
    Returns
    -----------------
    a boolean value on whether the cost function has converged or not. 
    
    
    """

    t1 = np.average(cost_list[:window], axis=0)
    t2 = np.average(cost_list[window:], axis=0)
  

    test = (np.linalg.norm(t1 - t2) / np.linalg.norm(t1))


    return test <= tolerance	    
 ##################################################################################   

