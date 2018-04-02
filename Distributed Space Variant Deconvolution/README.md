# Distributed Space Variant Deconvolution

## Introduction

This folder implements the Dedale distributed learning architecture for solving the space variant deconvolution of a large-scale stack of galaxy survey images.

The original Python library (for standalone execution over a small number of stacked images) is available [here](https://github.com/sfarrens/sf_deconvolve).

## Prerequisities and dependencies

* A Spark-complian cluster, according to the guidelines available [here](../README.md)

* For running the space variant deconvolution modules over the cluster, each of the cluster nodes (master and slaves) should have installed:
	- [Numpy](http://www.numpy.org/). Tested with vestion 1.13.3
	- [Scipy](http://www.scipy.org/). Tested with vestion 1.0.0
	- [Astropy](http://www.astropy.org/). Tested with version 2.0.2 
	- [Future](https://pypi.python.org/pypi/future). Tested with version 0.16.0
	- [iSap / Sparse2D library](http://www.cosmostat.org/software/isap) and [cfitsio](https://heasarc.gsfc.nasa.gov/fitsio/fitsio.html) libraries. Tested with iSAP version 3.1. The Sparse2D library should be compiled C++ modules. For a cluster with the 64-bit Ubuntu 16.04 LTS configuration, the precompiled libraries available [here](../docs/useful/) may be used. 
	For configuring the ISAP library on each terminal of the cluster, add the Sparse2D executables to the $PATH system variable. For example if the ISAP library is compiled at `/home/user/isap`, then:
		- Open the .profile file at `/home/user/`. 
		- Define the isap location: `ISAP="$HOME/isap`
		- Append the `$PATH` variable: `PATH="$ISAP/cxx/sparse2d/bin:$PATH"`
		- Save and close the .profile file. Log out and log back in for the change on the `$PATH` to take effect.

## Deployment on cluster

For deploymet over the cluster:

* Download the contents of this subfolder at the master node at a location with read/write/execute permissions. For the purposes of this guide In this guide, the preselected folder is `/home/user/ds_psf`.

* Compress the `lib` folder,  which contains the deconvolution modules (including both standalone and distributed execution), into `lib.zip`.

* Compress the `sf_tools` folder, which contains original optimisation and analysis modules (taken from [here](https://github.com/sfarrens/sf_deconvolve)) into `sf_tools.zip`


## Execution

The main python script for execution is the `dl_psf_deconvolve.py`. Nevertheless, all input parameters for execution can be defined at the `runexper.sh`. 

The format of each entry at the `runexper.sh` is the following:

`$SPARK`/bin/spark-submit --master spark://`<IP of master node>`:7077 --py-files lib.zip,sf_tools.zip  dl_psf_deconvolve.py -i `<input stack of noisy data>`.npy -p `<input psf>`.npy --mode `` --n_iter `<number of optimization iterations>` --pn `<number of blocks per RDD>`  > `<application log file>`.txt
mv log.out `<spark log file>`.out

where:
*  `$SPARK`: the folder of the spark build version at the master node (e.g., `/usr/local/spark`)

* `<IP of master node>`: the IP of the master node

*  `<input stack of noisy data>`.npy is the location and name of the input data in npy format (e.g., example_data/100x41x41/example_image_stack)

*   `<input psf>`.npy is the is the location and name of the psf in npy format (e.g., example_data/100x41x41/example_psfs)

*   `<number of optimization iterations>`: is the maximum number of optimization iterations

*   `<number of blocks per RDD>`: is the number of data blocks for splitting the input data. In a typical cluster this number should be at least the double of total available CPU cores (for example if the cluster has 24 CPU cores, then `<number of blocks per RDD>` >=48) 



### Input data format


### Operational parameters


### Output data format


## Examples


### Optimization based on low-rank approximation


### Sparsity-based optimization


### 

## Reference Documents: 

* A. Panousopoulou, S. Farrens, S., Y. Mastorakis, J.L. Starck, P. Tsakalides, "A distributed learning architecture for big imaging problems in astrophysics," In 2017 25th European (pp. 1440-1444). IEEE.

* S.  Farrens,  F.M.  Ngole  Mboula,  and  J.-L.  Starck,  “Space variant deconvolution of galaxy survey images,”  Astronomy and 
& Astrophysics, vol. 601, A66 , 2017.
