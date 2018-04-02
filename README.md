# DEDALE Distributed Learning Toolbox (D4.3)

> Toolbox Main Authors: **Nancy Panousopoulou (FORTH-ICS), Samuel Farrens (CEA), Konstantina Fotiadou (FORTH-ICS), Greg Tsagkatakis (FORTH-ICS)**  
> Year: **2017-2018**   
> Corresponding Author Email: [apanouso@ics.forth.gr](mailto:apanouso@ics.forth.gr)
> Website: [https://github.com/dedale-fet](https://github.com/dedale-fet)  

This repository contains the Python toolbox for distributed sparsity-based learning architectures, along with benchmarking imaging test sets.


The toolbox implements the Dedale Distributed Learning Architecture for solving large-scale imaging problems, associated to:

* Space variant deconvolution of galaxy survey images (package: Distributed Space Variant Deconvolution)
* Hyperspectral and color image super resolution  (package: Distributed Sparce Coupled Dictionary Learning)

Please refer to the documentation in each sub-folder for more details on how to use the toolbox, deploy, and execute each application.

## Prerequisities for deploying and using the toolbox

The implementation of the Distributed Learning Architecture considers the use of the Apache Spark distributed computing framework.

The prerequisities for installing a Spark-compliant cluster over a set of working terminals are:

* Linux OS (tested with Ubuntu 16.04.3 LTS)

* Apache Spark. Tested the version 2.1.1 (pre-build with Apache Hadoop 2.7 and later), which is available for download [here](https://spark.apache.org/downloads.html).

* [Python](https://www.python.org/). Tested with version 2.7.12.

* Java. Tested with version 1.8.0



## Reference Documents: 

* K.  Fotiadou, G. Tsagkatakis, P. Tsakalides, "Linear Inverse Problems with Sparsity Constraints," DEDALE DELIVERABLE 3.1, 2016.  

* A. Panousopoulou, S. Farrens, S., Y. Mastorakis, J.L. Starck, P. Tsakalides, "A distributed learning architecture for big imaging problems in astrophysics," In 2017 25th European (pp. 1440-1444). IEEE.

* S.  Farrens,  F.M.  Ngole  Mboula,  and  J.-L.  Starck,  “Space variant deconvolution of galaxy survey images,”  Astronomy and 
& Astrophysics, vol. 601, A66 , 2017.













