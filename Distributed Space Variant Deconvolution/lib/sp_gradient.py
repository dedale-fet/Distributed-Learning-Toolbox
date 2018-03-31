# -*- coding: utf-8 -*-
"""GRADIENT CLASSES

This module contains classses for defining PSF deconvolution specific gradients. 

This is the modified version of different classes for the needs of the distributed learning archiecture

:Author: Samuel Farrens <samuel.farrens@gmail.com>, modified by Nancy Panousopoulou@FORTH-ICS (apanouso@ics.forth.gr)

:Version: 1.0

:Initial Release: 19/07/2017

:Revised Version: 13/12/2017
"""

import numpy as np
from sf_tools.signal.gradient import GradBasic
from sf_tools.math.matrix import PowerMethod
from sf_tools.base.transform import cube2matrix, matrix2cube
from sf_tools.image.convolve import psf_convolve, convolve_stack
from sf_tools.image.shape import shape_project


##Gradient classes##
"""GRADIENT CLASSES


This module contains classses for defining PSF deconvolution specific
gradients.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 19/07/2017

"""

class GradPSF(PowerMethod):
    """Gradient class for PSF convolution

    This class defines the operators for a fixed or object variant PSF

    Parameters
    ----------
    data : np.ndarray
        Input data array, an array of 2D observed images (i.e. with noise)
    psf : np.ndarray
        PSF, a single 2D PSF or an array of 2D PSFs
    psf_type : str {'fixed', 'obj_var'}
        PSF type (defualt is 'fixed')

    Notes
    -----
    The properties of `PowerMethod` are not inherited in this class

    """

    def __init__(self, data, psf, psf_type='fixed'):

        self._y = np.copy(data)
        self._psf = np.copy(psf)
        self._psf_type = psf_type

        PowerMethod.__init__(self, lambda x: self.Ht_op(self.H_op(x)),
                             self._y.shape, auto_run=False)

    def H_op(self, x):
        """H matrix operation

        This method calculates the action of the matrix H on the input data, in
        this case the convolution of the the input data with the PSF

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        Returns
        -------
        np.ndarray result

        """

        return psf_convolve(x, self._psf, psf_rot=False,
                            psf_type=self._psf_type)

    def Ht_op(self, x):
        """Ht matrix operation

        This method calculates the action of the transpose of the matrix H on
        the input data, in this case the convolution of the the input data with
        the rotated PSF

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        Returns
        -------
        np.ndarray result

        """

        return psf_convolve(x, self._psf, psf_rot=True,
                            psf_type=self._psf_type)

    def _calc_grad(self, x):

        return self.Ht_op(self.H_op(x) - self._y)


class SP_GradKnownPSF(GradPSF):
    """Gradient class for a known PSF

    This class calculates the gradient when the PSF is known

    Parameters
    ----------
    data : np.ndarray
        Input data array, an array of 2D observed images (i.e. with noise)
    psf : np.ndarray
        PSF, a single 2D PSF or an array of 2D PSFs
    psf_type : str {'fixed', 'obj_var'}
        PSF type (defualt is 'fixed')

    Notes
    -----
    The properties of `GradPSF` are inherited in this class

    """

    def __init__(self, data, psf, psf_type='fixed'):

        self.grad_type = 'psf_known'
        super(SP_GradKnownPSF, self).__init__(data, psf, psf_type)

    def get_grad(self, x):
        """Get the gradient at the given iteration

        This method calculates the gradient value from the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        """

        self.grad = self._calc_grad(x)

