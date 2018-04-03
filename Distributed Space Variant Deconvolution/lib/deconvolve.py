# -*- coding: utf-8 -*-

"""PSF DECONVOLUTION MODULE

This module deconvolves a set of galaxy images with a known object-variant PSF using the distributed learning architecture (master module)

:Author: Samuel Farrens <samuel.farrens@gmail.com> (deconvolution modules), Nancy Panousopoulou@FORTH-ICS (apanous@ics.forth.gr) distributed learning library


:Initial Release: 23/10/2017 
:Modification: 13/12/2017

"""

from __future__ import print_function
from builtins import range, zip
from scipy.linalg import norm
from sf_tools.signal.optimisation import *
from sf_tools.math.stats import sigma_mad
from sf_tools.signal.cost import costObj
from sf_tools.signal.linear import *
from sf_tools.signal.proximity import *
from sf_tools.signal.reweight import cwbReweight
from sf_tools.signal.wavelet import filter_convolve, filter_convolve_stack
from gradient import *
from cost import sf_deconvolveCost

###################################
# Libraries needed for the distributed learning library
#@nancypan-FORTHICS

import time
from sp_operators import *
from sp_gradient import *

from  pyspark import SparkContext


from sympy.logic.boolalg import false
from sf_tools.signal import noise
import scipy.io as sio




########################################################################
#@nancypan-FORTHICS
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

########################################################################
def calc_spec_rad_rdd(rdd_x, rdddata2, data_shape, data_type='obj_var', tolerance=1e-6, max_iter=10):
	
	"""Calculate the spectral radius over the Resilient Distributed 
	Databases representing the noisy and the psf data using the power method

    Input arguments
    ----------
    rdd_x: RDD
    	   The RDD representation of the noisy data
    
    rdddata2: RDD
    		  The RDD representation of the PSF data
    
    data_shape: tuple
    			The shape of the input data
    
    data_type: str{'fixed', 'obj_var'}
    		   The type of PSF --needed for convolution -- fixed or spatially variant.
    
    tolerance:  float
    			Tolerance for convergence 
    
    max_iter: integer
    		  The maximum number of iterations for calculating the spectral radius
    
    Returns
    -------
    The value of the spectral radius.
    """
    
	for i in range(max_iter):    
		print("current iteration:")
		print(i)

		x_old_norm = np.linalg.norm(np.reshape(np.array(rdd_x.collect()), data_shape))
		rdd_x = rdd_x.zip(rdddata2).cache()

		rdd_x = rdd_x.map(lambda x:sp_psf_convolve([x for k in range(0,1)],rot=False, data_type=data_type))

		rdd_x = rdd_x.zip(rdddata2).cache()

		rdd_x = rdd_x.map(lambda x:sp_psf_convolve([x for k in range(0,1)], rot=True, data_type=data_type))

		x_new = np.array(rdd_x.collect()) / x_old_norm
		if(np.abs(np.linalg.norm(x_new) - x_old_norm) < tolerance):
			print (' - Power Method converged after %d iterations!' % (i + 1))
		 	break

		elif i == max_iter - 1:
			print (' - Power Method did not converge after %d iterations!' % max_iter)
        
		print("############################################")
      
	
	return np.linalg.norm(x_new)
########################################################################

########################################################################
def sp_set_grad_op(data, psf, rdd_data, rdd_psf, **kwargs):
	
    """Set the gradient operator for the RDD

    This method defines the gradient operator class to use and add an 
    instance to the keyword arguments.

    Input Arguments
    ----------
    data : np.ndarray
        Input noisy data (3D array)
    psf : np.ndarray
        PSF data (2D or 3D array)
    
    rdd_data: RDD
    		  the RDD representation of the input noisy data
        
    rdd_psf: RDD
    		the RDD representation of the PSF data
    
     
    Returns
    -------
    kwargs: dict 
    		The spectral radius for the gradient operator in the updated keyword arguments

    """
    
    # Set the gradient operator
    if kwargs['grad_type'] == 'psf_known':
     
        kwargs['grad_op'] = SP_GradKnownPSF(data, psf,psf_type=kwargs['psf_type'])
        
        kwargs['grad_op'].spec_rad = calc_spec_rad_rdd(rdd_data, rdd_psf, data.shape, data_type='obj_var')
        
    else:
        print("The grad type provided is not currently supported for distributed spectral radius calculation. The grad_type supported are: psf_known")
        print("System will now exit...")
        import sys
        sys.exit(1)
    
    
    print(' - Spectral Radius:', kwargs['grad_op'].spec_rad)
    
    if 'log' in kwargs:
        kwargs['log'].info(' - Spectral Radius: ' + str(kwargs['grad_op'].spec_rad))

    return kwargs
########################################################################


########################################################################
def sp_set_sparse_weights(data_shape, **kwargs):
    
    """Set the sparsity weights -- considering the sparse mode only.

    This method defines the weights for thresholding in the sparse domain and
    add them to the keyword arguments. It additionally defines the shape of the
    dual variable.

    Input Arguments
    ----------
    data_shape : tuple
        Shape of the input data array
    
    Returns
    -------
    kwargs: dict 
    		The updated keyword arguments

    """
    
    # Set the shape of the dual variable
    dual_shape = ([kwargs['wavelet_filters'].shape[0]] + list(data_shape))
    dual_shape[0], dual_shape[1] = dual_shape[1], dual_shape[0]
    kwargs['dual_shape'] = dual_shape

    return kwargs
########################################################################

########################################################################
def sp_condat_optimization_sparse(rdd_data, rdd_psf, partitions_num, data_shape, **kwargs):

    """The main optimization function using CONDAT for the sparsity optimization mode.

    Input Arguments
    ----------
    rdd_data : RDD
    		   The RDD representation of the input noisy data (3D array)
        
    rdd_psf : RDD
    		  The RDD representation of the PSF data (3D array)
        
    partitions_num : integer
    				 The number of partitions of the RDD bundle component
    
    data_shape : tuple
    		     The shape of the input data 
    
    kwargs : dict
    		 the operational parameters for the optimization.
     
    Returns
    -------
    
    cost_list : list
    			The values of the cost function
    time_all  : list 
    			The time elapsed per iteration
    time_tot  : float
    			The total time of execution for the optimization
    			
    primal   :np.ndarray
    		   The resulting primal optimization variable
    dual	 :np.ndarray
    			The resulting dual optimization variable		   	
    """
    
    
    wf = kwargs['wavelet_filters']
    print(">>>>>>>>>>>>>>.")
    print(wf.shape)
    print(data_shape)
    print("<<<<<<<<<<<")
    
    wv_thr_factor = kwargs['wave_thresh_factor']
    noise_est = kwargs['noise_est']
    
    # Convolve the PSF with the wavelet filters
    
    if kwargs['psf_type'] == 'fixed':
    
        print("This lowr_type is not currently supported. The types supported are: standard")
        print("System will now exit")
        import sys
        sys.exit(1)
    
        
    # Primal Operator for condat optimization
    primal = np.ones(data_shape)
    rdd_primal = sc.parallelize(primal, partitions_num)#.cache()
   

    # Dual Operator for condat optimization
    dual = np.ones(kwargs['dual_shape'])
    rdd_dual =  sc.parallelize(dual, partitions_num)
      
    # Combine all rdds into a single one (for the bundle component)
    rdd_in = rdd_data.zip(rdd_psf).zip(rdd_primal).zip(rdd_dual).cache()

    
    cost_list = []
    time_all =[]
    
    #get the optimization parameters 
    rho = kwargs['relax']
    sigma = kwargs['condat_sigma']
    tau = kwargs['condat_tau']
    
    window = kwargs['cost_window']
    convergence = kwargs['convergence']
    wf = kwargs['wavelet_filters']
    n_iters = kwargs['n_iter']
    ttime2 = time.time()
    print('entering optimization...>>>>>>:')
    
    for i in range(n_iters): 
        # run the update (Calculate x,y)
        print(i)
        ttime3 = time.time()
        
        #update the orimal-dual variables over the distributed architecture
        rdd_in = rdd_in.map(lambda x:sp_runupdate_sparse(x,wf, rho,sigma,tau,wv_thr_factor, noise_est)).cache()
        
		#calculate the cost function over the distributed architecture
        d1cost = rdd_in.map(lambda x:sp_calc_cost_sparse(x,wf,wv_thr_factor, noise_est))
        
        #keep the cost value
        cost_list.append(d1cost.reduce(lambda x,y: x+y))
        print('time elapsed:')
        print(time.time()-ttime3)
        time_all.append(time.time()-ttime3)        
        if (i+1) % (2 * window):
            continue
    
        else:
         
            print(np.log10((np.array(cost_list))))
            
            #check the convergence of the cost value
            converged = sp_check_convergence(cost_list, window, convergence)
            
            if converged:
                print(' - Converged!')
                break    
           
            #clean up your system
            sc._jvm.System.gc() 
            
    print('>>>>total time elapsed>>>>>>.: ')
    time_tot = time.time()-ttime2 
    print(time_tot)
    
    fin_data = rdd_in.collect()
    
	#Optional: save results to hdfs file
	#folder2results = 'results_data_' + str(partitions_num) + '.txt'
    
	#rdd_in.saveAsTextFile(folder2results)
	#extract the primal & dual opt. variables
    fin_data = rdd_in.collect()
    
    tmp1, dual = zip(*zip(*zip(*zip(*zip(*fin_data)))))

    tmp2, primal = zip(*zip(*zip(zip(*tmp1))))
    
    
    return [cost_list, time_all,time_tot, np.squeeze(np.array(primal)), np.array(dual)] 
###############################################################################

###############################################################################
def sp_condat_optimization_lowr(rdd_data, rdd_psf, partitions_num, data_shape, **kwargs):
	"""The main optimization function using CONDAT for the low-rank optimization mode.

	Input Arguments
	----------
	rdd_data : RDD
			   The RDD representation of the input noisy data (3D array)
		
	rdd_psf : RDD
			  The RDD representation of the PSF data (3D array)
		
	partitions_num : integer
					 The number of partitions of the RDD bundle component
	
	data_shape : tuple
				 The shape of the input data 
	
	kwargs : dict
			 the operational parameters for the optimization.
	 
	Returns
	-------
	
	cost_list : list
				The values of the cost function
	time_all  : list 
				The time elapsed per iteration
	time_tot  : float
				The total time of execution for the optimization
	primal   :np.ndarray
    		   The resulting primal optimization variable
    dual	 :np.ndarray
    			The resulting dual optimization variable
    """


	# Primal Operator for condat optimization
	primal = np.ones(data_shape)
	rdd_primal = sc.parallelize(primal, partitions_num)#.cache()
   

    # Dual Operator for condat optimization
	dual = np.ones(data_shape)
	rdd_dual =  sc.parallelize(dual, partitions_num)#.cache()#sp_weight_est.map(lambda x:0.5*np.ones(dual_shape))
      
    # Combine all rdds into a single one (for the bundle component)
	rdd_in = rdd_data.zip(rdd_psf).zip(rdd_primal).zip(rdd_dual).cache()
                
	cost_list = []
	time_all =[]
    
    #get the optimization parameters
	rho = kwargs['relax']
	sigma = kwargs['condat_sigma']
 	tau = kwargs['condat_tau']
 	thres_type=kwargs['lowr_thresh_type']
 	window = 2#kwargs['cost_window']
 	convergence = kwargs['convergence']
 	lamb = kwargs['lambda']
 	n_iters = kwargs['n_iter']
 	ttime2 = time.time()
 	print('entering optimization...>>>>>>:')
    
	for i in range(n_iters): 

		print(i)
		ttime3 = time.time()
		# calculate the updated primal, dual optimization variables over the distributed architecture
		rdd_in = rdd_in.map(lambda x:sp_runupdate_lowr(x, lamb, rho, sigma, tau, thres_type = thres_type)).cache()

		#calculate the cost value over the distributed architecture
		d1cost = rdd_in.map(lambda x:sp_calc_cost_lowr(x,lamb))
		
		#keep the cost value
		cost_list.append(d1cost.reduce(lambda x,y: x+y))
		
		print('time elapsed:')
		print(time.time()-ttime3)
		time_all.append(time.time()-ttime3)        
		if (i+1) % (2 * window):
			continue
		
		else:
			print(np.log10((np.array(cost_list))))
        
        
            #print('time elapsed:')
            #check the convergence of the cost function
			converged = sp_check_convergence(cost_list, window, convergence)
            
			if converged:
				print(' - Converged!')
				break    

			#clear up your system
			sc._jvm.System.gc() 
	
	print('>>>>total time elapsed:>>>>>>.: ')
	time_tot = time.time()-ttime2 
 	print(time_tot)

	#Optional:  save to hdfs file...(save to File or zip with index action)....
	#folder2results = 'psf_results_smalldata_' + str(partitions_num) + '.txt'
	
	#rdd_in.saveAsTextFile(folder2results)

    #extract the primal & dual opt. variables
	fin_data = rdd_in.collect()

	tmp1, dual = zip(*zip(*zip(*zip(*zip(*fin_data)))))

	tmp2, primal = zip(*zip(*zip(zip(*tmp1))))

	return [cost_list, time_all,time_tot, np.squeeze(np.array(primal)), np.array(dual)] 

####################################################################################


#@nancypan-FORTHICS
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
####################################################################################


####################################################################################
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
##routines for standalone execution.

#####################################################################################
def set_noise(data, **kwargs):
    """Set the noise level

    This method calculates the noise standard deviation using the median
    absolute deviation (MAD) of the input data and adds it to the keyword
    arguments.

    Parameters
    ----------
    data : np.ndarray
        Input noisy data (3D array)

    Returns
    -------
    dict Updated keyword arguments

    """

    # It the noise is not already provided calculate it using the MAD
    if isinstance(kwargs['noise_est'], type(None)):
        kwargs['noise_est'] = sigma_mad(data)

    print(' - Noise Estimate:', kwargs['noise_est'])
    if 'log' in kwargs:
        kwargs['log'].info(' - Noise Estimate: ' + str(kwargs['noise_est']))

    return kwargs
##################################################################################


##################################################################################
def set_grad_op(data, psf, **kwargs):
    """Set the gradient operator

    This method defines the gradient operator class to use and add an instance
    to the keyword arguments.

    Parameters
    ----------
    data : np.ndarray
        Input noisy data (3D array)
    psf : np.ndarray
        PSF data (2D or 3D array)

    Returns
    -------
    dict Updated keyword arguments

    """
    
    # Set the gradient operator
    if kwargs['grad_type'] == 'psf_known':
     
        kwargs['grad_op'] = GradKnownPSF(data, psf,
                                         psf_type=kwargs['psf_type'])

    elif kwargs['grad_type'] == 'psf_unknown':
        kwargs['grad_op'] = GradUnknownPSF(data, psf, Positive(),
                                           psf_type=kwargs['psf_type'],
                                           beta_reg=kwargs['beta_psf'],
                                           lambda_reg=kwargs['lambda_psf'])

    elif kwargs['grad_type'] == 'shape':
        kwargs['grad_op'] = GradShape(data, psf, psf_type=kwargs['psf_type'],
                                      lambda_reg=kwargs['lambda_shape'])

    elif kwargs['grad_type'] == 'none':
        kwargs['grad_op'] = GradNone(data, psf, psf_type=kwargs['psf_type'])
    
    
    
    
    print(' - Spectral Radius:', kwargs['grad_op'].spec_rad)
    if 'log' in kwargs:
        kwargs['log'].info(' - Spectral Radius: ' +
                           str(kwargs['grad_op'].spec_rad))

    return kwargs
##########################################################################################



###########################################################################################
def set_linear_op(data, **kwargs):
    """Set the gradient operator

    This method defines the gradient operator class to use and add an instance
    to the keyword arguments. It additionally add the l1 norm of the linear
    operator and the wavelet filters (if used) to the kwagrs.

    Parameters
    ----------
    data : np.ndarray
        Input noisy data (3D array)

    Returns
    -------
    dict Updated keyword arguments

    ToDo
    ----
    - Clean up wavelet_filters and l1norm

    """

    # Set the options for mr_transform (for sparsity)
    if kwargs['mode'] in ('all', 'sparse'):
        wavelet_opt = ['-t ' + kwargs['wavelet_type']]

    # Set the linear operator
    if kwargs['mode'] == 'all':
        kwargs['linear_op'] = LinearCombo([Wavelet(data, wavelet_opt),
                                          Identity()])
        kwargs['wavelet_filters'] = kwargs['linear_op'].operators[0].filters
        kwargs['linear_l1norm'] = kwargs['linear_op'].operators[0].l1norm

    elif kwargs['mode'] in ('lowr', 'grad'):
        kwargs['linear_op'] = Identity()
        kwargs['linear_l1norm'] = kwargs['linear_op'].l1norm

    elif kwargs['mode'] == 'sparse':
        kwargs['linear_op'] = Wavelet(data, wavelet_opt)
        kwargs['wavelet_filters'] = kwargs['linear_op'].filters
        kwargs['linear_l1norm'] = kwargs['linear_op'].l1norm

    return kwargs
#################################################################################################


#################################################################################################
def set_sparse_weights(data_shape, psf, **kwargs):
    """Set the sparsity weights

    This method defines the weights for thresholding in the sparse domain and
    add them to the keyword arguments. It additionally defines the shape of the
    dual variable.

    Parameters
    ----------
    data_shape : tuple
        Shape of the input data array
    psf : np.ndarray
        PSF data (2D or 3D array)

    Returns
    -------
    dict Updated keyword arguments

    """

    # Convolve the PSF with the wavelet filters
    if kwargs['psf_type'] == 'fixed':

        filter_conv = (filter_convolve(np.rot90(psf, 2),
                       kwargs['wavelet_filters']))

        filter_norm = np.array([norm(a) * b * np.ones(data_shape[1:])
                                for a, b in zip(filter_conv,
                                kwargs['wave_thresh_factor'])])

        filter_norm = np.array([filter_norm for i in
                                range(data_shape[0])])

    else:

        filter_conv = (filter_convolve_stack(np.rot90(psf, 2),
                       kwargs['wavelet_filters']))

        filter_norm = np.array([[norm(b) * c * np.ones(data_shape[1:])
                                for b, c in zip(a,
                                kwargs['wave_thresh_factor'])]
                                for a in filter_conv])

    # Define a reweighting instance
    kwargs['reweight'] = cwbReweight(kwargs['noise_est'] * filter_norm)

    # Set the shape of the dual variable
    dual_shape = ([kwargs['wavelet_filters'].shape[0]] + list(data_shape))
    dual_shape[0], dual_shape[1] = dual_shape[1], dual_shape[0]
    kwargs['dual_shape'] = dual_shape

    return kwargs
##############################################################################################

##############################################################################################
def set_condat_param(**kwargs):
    """Set the Condat-Vu parameters

    This method sets the values of tau and sigma in the Condat-Vu proximal-dual
    splitting algorithm if not already provided. It additionally checks that
    the combination of values will lead to convergence.

    Returns
    -------
    dict Updated keyword arguments

    """

    # Define a method for calculating sigma and/or tau
    def get_sig_tau():
        return 1.0 / (kwargs['grad_op'].spec_rad + kwargs['linear_l1norm'])

    # Calulate tau if not provided
    if isinstance(kwargs['condat_tau'], type(None)):
        kwargs['condat_tau'] = get_sig_tau()

    # Calculate sigma if not provided
    if isinstance(kwargs['condat_sigma'], type(None)):
        kwargs['condat_sigma'] = get_sig_tau()

    print(' - tau:', kwargs['condat_tau'])
    print(' - sigma:', kwargs['condat_sigma'])
    print(' - rho:', kwargs['relax'])
    if 'log' in kwargs:
        kwargs['log'].info(' - tau: ' + str(kwargs['condat_tau']))
        kwargs['log'].info(' - sigma: ' + str(kwargs['condat_sigma']))
        kwargs['log'].info(' - rho: ' + str(kwargs['relax']))

    # Test combination of sigma and tau
    sig_tau_test = (1.0 / kwargs['condat_tau'] - kwargs['condat_sigma'] *
                    kwargs['linear_l1norm'] ** 2 >=
                    kwargs['grad_op'].spec_rad / 2.0)

    print(' - 1/tau - sigma||L||^2 >= beta/2:', sig_tau_test)
    if 'log' in kwargs:
        kwargs['log'].info(' - 1/tau - sigma||L||^2 >= beta/2: ' +
                           str(sig_tau_test))

    return kwargs

############################################################################################

#############################################################################################
def get_lambda(n_images, p_pixel, sigma, spec_rad):
    """Get lambda value

    This method calculates the singular value threshold for low-rank
    regularisation

    Parameters
    ----------
    n_images : int
        Total number of images
    p_pixel : int
        Total number of pixels
    sigma : float
        Noise standard deviation
    spec_rad : float
        The spectral radius of the gradient operator

    Returns
    -------
    float Lambda value

    """

    return sigma * np.sqrt(np.max([n_images + 1, p_pixel])) * spec_rad
################################################################################################


#################################################################################################
def set_lowr_thresh(data_shape, **kwargs):
    """Set the low-rank threshold

    This method sets the value of the low-rank singular value threshold.

    Parameters
    ----------
    data_shape : tuple
        Shape of the input data array

    Returns
    -------
    dict Updated keyword arguments

    """

    if kwargs['lowr_type'] == 'standard':
        kwargs['lambda'] = (kwargs['lowr_thresh_factor'] *
                            get_lambda(data_shape[0], np.prod(data_shape[1:]),
                            kwargs['noise_est'], kwargs['grad_op'].spec_rad))

    elif kwargs['lowr_type'] == 'ngole':
        kwargs['lambda'] = (kwargs['lowr_thresh_factor'] * kwargs['noise_est'])

    print(' - lambda:', kwargs['lambda'])
    if 'log' in kwargs:
        kwargs['log'].info(' - lambda: ' + str(kwargs['lambda']))

    return kwargs
##################################################################################################

##################################################################################################
def set_primal_dual(data_shape, **kwargs):
    """Set primal and dual variables

    This method sets the initial values of the primal and dual variables

    Parameters
    ----------
    data_shape : tuple
        Shape of the input data array

    Returns
    -------
    dict Updated keyword arguments

    """

    # Set the initial values of the primal variable if not provided
    if isinstance(kwargs['primal'], type(None)):
        kwargs['primal'] = np.ones(data_shape)

    # Set the initial values of the dual variable
    if kwargs['mode'] == 'all':
        kwargs['dual'] = np.empty(2, dtype=np.ndarray)
        kwargs['dual'][0] = np.ones(kwargs['dual_shape'])
        kwargs['dual'][1] = np.ones(data_shape)

    elif kwargs['mode'] in ('lowr', 'grad'):
        kwargs['dual'] = np.ones(data_shape)

    elif kwargs['mode'] == 'sparse':
        kwargs['dual'] = np.ones(kwargs['dual_shape'])

    print(' - Primal Variable Shape:', kwargs['primal'].shape)
    print(' - Dual Variable Shape:', kwargs['dual'].shape)
    print(' ' + '-' * 70)
    if 'log' in kwargs:
        kwargs['log'].info(' - Primal Variable Shape: ' +
                           str(kwargs['primal'].shape))
        kwargs['log'].info(' - Dual Variable Shape: ' +
                           str(kwargs['dual'].shape))

    return kwargs
#########################################################################################################


##########################################################################################################
def set_prox_op_and_cost(data, **kwargs):
    """Set the proximity operators and cost function

    This method sets the proximity operators and cost function instances.

    Parameters
    ----------
    data : np.ndarray
        Input noisy data (3D array)

    Returns
    -------
    dict Updated keyword arguments

    """

    # Create a list of proximity operators
    kwargs['prox_op'] = []

    # Set the first operator as positivity contraint or simply identity
    if not kwargs['no_pos']:
        kwargs['prox_op'].append(Positive())

    else:
        kwargs['prox_op'].append(Identity())

    # Add a second proximity operator and set the corresponding cost function
    if kwargs['mode'] == 'all':

        kwargs['prox_op'].append(ProximityCombo(
                                 [Threshold(kwargs['reweight'].weights,),
                                  LowRankMatrix(kwargs['lambda'],
                                  thresh_type=kwargs['lowr_thresh_type'],
                                  lowr_type=kwargs['lowr_type'],
                                  operator=kwargs['grad_op'].Ht_op)]))

        cost_instance = (sf_deconvolveCost(data, grad=kwargs['grad_op'],
                         wavelet=kwargs['linear_op'].operators[0],
                         weights=kwargs['reweight'].weights,
                         lambda_lowr=kwargs['lambda'],
                         mode=kwargs['mode'],
                         positivity=not kwargs['no_pos'],
                         verbose=not kwargs['quiet']))

    elif kwargs['mode'] == 'lowr':

        kwargs['prox_op'].append(LowRankMatrix(kwargs['lambda'],
                                 thresh_type=kwargs['lowr_thresh_type'],
                                 lowr_type=kwargs['lowr_type'],
                                 operator=kwargs['grad_op'].Ht_op))

        cost_instance = (sf_deconvolveCost(data, grad=kwargs['grad_op'],
                         wavelet=None, weights=None,
                         lambda_lowr=kwargs['lambda'], mode=kwargs['mode'],
                         positivity=not kwargs['no_pos'],
                         verbose=not kwargs['quiet']))

    elif kwargs['mode'] == 'sparse':

        kwargs['prox_op'].append(Threshold(kwargs['reweight'].weights))

        cost_instance = (sf_deconvolveCost(data, grad=kwargs['grad_op'],
                         wavelet=kwargs['linear_op'],
                         weights=kwargs['reweight'].weights,
                         lambda_lowr=None,
                         mode=kwargs['mode'],
                         positivity=not kwargs['no_pos'],
                         verbose=not kwargs['quiet']))

    elif kwargs['mode'] == 'grad':

        kwargs['prox_op'].append(Identity())

        cost_instance = (sf_deconvolveCost(data, grad=kwargs['grad_op'],
                         wavelet=None, weights=None,
                         lambda_lowr=None, mode=kwargs['mode'],
                         positivity=not kwargs['no_pos'],
                         verbose=not kwargs['quiet']))

    kwargs['cost_op'] = (costObj(cost_instance,
                         tolerance=kwargs['convergence'],
                         cost_interval=kwargs['cost_window'],
                         plot_output=kwargs['output'],
                         verbose=not kwargs['quiet']))

    return kwargs
###################################################################################################


####################################################################################################
def set_optimisation(**kwargs):
    """Set the optimisation technique

    This method sets the technique used for opttimising the problem

    Returns
    -------
    dict Updated keyword arguments

    """

    # Initalise an optimisation instance
    if kwargs['opt_type'] == 'fwbw':
        kwargs['optimisation'] = (ForwardBackward(kwargs['primal'],
                                  kwargs['grad_op'], kwargs['prox_op'][1],
                                  kwargs['cost_op'], auto_iterate=False))

    elif kwargs['opt_type'] == 'condat':
        print("test1<<<<<<<<<<<<<<<<<")
        kwargs['optimisation'] = (Condat(kwargs['primal'], kwargs['dual'],
                                  kwargs['grad_op'], kwargs['prox_op'][0],
                                  kwargs['prox_op'][1], kwargs['linear_op'],
                                  kwargs['cost_op'], rho=kwargs['relax'],
                                  sigma=kwargs['condat_sigma'],
                                  tau=kwargs['condat_tau'],
                                  auto_iterate=False))
        print(">>>>>>>>>>>>>>>>>>>>test2")

    elif kwargs['opt_type'] == 'gfwbw':
        kwargs['optimisation'] = (GenForwardBackward(kwargs['primal'],
                                  kwargs['grad_op'], kwargs['prox_op'],
                                  lambda_init=1.0, cost=kwargs['cost_op'],
                                  weights=[0.1, 0.9],
                                  auto_iterate=False))

    return kwargs
#############################################################################################

##############################################################################################
def perform_reweighting(**kwargs):
    """Perform reweighting

    This method updates the weights used for thresholding in the sparse domain

    Returns
    -------
    dict Updated keyword arguments

    """

    # Loop through number of reweightings
    for i in range(kwargs['n_reweights']):

        print(' - REWEIGHT:', i + 1)
        print('')

        # Generate the new weights following reweighting persctiption
        kwargs['reweight'].reweight(kwargs['linear_op'].op(
                                    kwargs['optimisation'].x_new)[0])

        # Perform optimisation with new weights
        kwargs['optimisation'].iterate(max_iter=kwargs['n_iter'])

        print('')
###################################################################################################

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
###################################################################################################


###################################################################################################
#Main Function for Image Deconvolution
def spark_run(data, psf, spark_variable = False, partitions_num = 1,iter_num=1,**kwargs):
    """Run deconvolution either in a stand-alone mode or over the distributed learning architecture

    This method initialises the operator classes and runs the optimisation
    algorithm

    Parameters
    ----------
    data : np.ndarray
        Input data array, an array of 2D images
    psf : np.ndarray
        Input PSF array, a single 2D PSF or an array of 2D PSFs
        
    spark_variable: boolean
    				Activating (True) the calculation of PSF deconvolution (sparse or low-rank) over the distributed learning architecture.
    
    
    partitions_num: integer
    				the number of partitions for the representation of the input data and psf as resilient distributed databases
    
    
    iter_num: integer
    		  the number of independent experiment

    Returns
    -------
    np.ndarray decconvolved data

    """
    
    
    time1 = time.time()
    
    if spark_variable:
    	global sc
    	sc = SparkContext()
    
     	#parallelize the data
        rdd_data = sc.parallelize(data, partitions_num).cache()
    	#parallelize the psf
        rdd_psf = sc.parallelize(psf, partitions_num).cache()
    
    
    
    # SET THE NOISE ESTIMATE
    kwargs = set_noise(data, **kwargs)

    # SET THE GRADIENT OPERATOR
    if not spark_variable:
        kwargs = set_grad_op(data, psf, **kwargs)
    else:
    	
        rdd_x = sc.parallelize(np.random.random(data.shape), partitions_num).cache()
        kwargs = sp_set_grad_op(data,psf, rdd_x, rdd_psf, **kwargs)
        rdd_x.unpersist()
            
    # SET THE LINEAR OPERATOR
    kwargs = set_linear_op(data, **kwargs)

    
    # SET THE WEIGHTS IN THE SPARSE DOMAIN
    
    if kwargs['mode'] in ('all', 'sparse'):
        ##@nancypan: distribute...
        if not spark_variable:
            kwargs = set_sparse_weights(data.shape, psf, **kwargs)
            #print(np.array(kwargs['reweight'].weights))
        else:
          
            kwargs  = sp_set_sparse_weights(data.shape,**kwargs)
            
            
            
    # SET THE CONDAT-VU PARAMETERS
    if kwargs['opt_type'] == 'condat':
        kwargs = set_condat_param(**kwargs)

    # SET THE LOW-RANK THRESHOLD
    if kwargs['mode'] in ('all', 'lowr'):
        kwargs = set_lowr_thresh(data.shape, **kwargs)

    # SET THE INITIAL PRIMAL AND DUAL VARIABLES
    if not spark_variable:
    	#standalone execution
        kwargs = set_primal_dual(data.shape, **kwargs)
        # SET THE PROXIMITY OPERATORS AND THE COST FUNCTION
        kwargs = set_prox_op_and_cost(data, **kwargs)
        
        # SET THE OPTIMISATION METHOD
        kwargs = set_optimisation(**kwargs)

        # PERFORM OPTIMISATION
        ##@nancypan: replace.
        kwargs['optimisation'].iterate(max_iter=kwargs['n_iter'])
        
    
    
    else:
    #distributed execution 
        
        if kwargs['opt_type'] == 'condat':
            if kwargs['mode'] == 'lowr':
            	
                ## no need for weights.
                if kwargs['lowr_type'] == 'standard':
                	
                    [cost_list, time_all, time_tot, xfinal, yfinal]= sp_condat_optimization_lowr(rdd_data, rdd_psf, partitions_num, data.shape, **kwargs)
                    print('done with optimization - now saving....!')
                    #store the results of execution into a matfile for further post-processing.
                    
                    mat2save =  './results' + '_' + kwargs['mode'] + '_' +str(data.shape[0]) + '_' + str(partitions_num) + '_' + str(iter_num) + '_n'+ str(kwargs['n_iter'])  +'.mat'  
                    
                    sio.savemat(mat2save, {'cost':cost_list, 'time_iters': time_all, 'time_totopt':time_tot,'time_tot':time.time()-time1, 'primal_var': xfinal, 'dual_var': yfinal})
                    
                else:
                    print("This lowr_type is not currently supported. The types supported are: standard")
                    print("System will now exit")
                    import sys
                    sys.exit(1)
                
            elif kwargs['mode'] == 'sparse':
                
                                         
                 [cost_list, time_all, time_tot, xfinal, yfinal] = sp_condat_optimization_sparse(rdd_data, rdd_psf, partitions_num, data.shape, **kwargs)
                 
                 #store the results of execution into a matfile for further post-processing.
                 print('done with optimization - now saving....!')
                 mat2save =  './results' + '_' + kwargs['mode'] + '_' +str(data.shape[0]) + '_' +str(partitions_num) + '_' + str(iter_num) + '_n'+ str(kwargs['n_iter']) +  '.mat'  
                 sio.savemat(mat2save, {'cost':cost_list, 'time_iters': time_all, 'time_totopt':time_tot, 'time_tot':time.time()-time1, 'primal_var': xfinal, 'dual_var': yfinal})
                 
                   
            else:
                print("This mode is not currently supported. The modes supported are: lowr, sparse")
                print("System will now exit")
                import sys
                sys.exit(1)    
        else:
            print("The optimization method is not currently supported. The optimization methods supported are: condat")
            print("System will now exit")
            import sys
            sys.exit(1)
                
    
    if spark_variable:    
     	sc.stop()
     	
                 
    # PLOT THE COST FUNCTION
    '''if not kwargs['no_plots']:
        kwargs['cost_op'].plot_cost()
    '''
    
    
    if not spark_variable: 
    # FINISH AND RETURN RESULTS
    	if 'log' in kwargs:
        	kwargs['log'].info(' - Final iteration number: ' +
                           str(kwargs['cost_op']._iteration))
         	kwargs['log'].info(' - Final log10 cost value: ' +
                           str(np.log10(kwargs['cost_op'].cost)))
          	kwargs['log'].info(' - Converged: ' + str(kwargs['optimisation'].converge))

      		primal_res = kwargs['optimisation'].x_final
	  		   
    		if kwargs['opt_type'] == 'condat':
         		dual_res = kwargs['optimisation'].y_final
    	  	else:
         		dual_res = None

		  	if kwargs['grad_type'] == 'psf_unknown':
		  		psf_res = kwargs['grad_op']._psf
		  	else:
		  		psf_res = None
		  		
		
       	return primal_res, dual_res, psf_res
    
    
    else:
    	#in the distributed execution we return in a mat file: 
    	#(a) the cost values in a list, 
    	#(b) the execution times, 
    	#(c) the final primal and dual optimization matrices
    	
    	return 1   	
    
###################################################################################################################

