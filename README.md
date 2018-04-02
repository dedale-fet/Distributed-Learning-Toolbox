# DEDALE Distributed Learning Toolbox (D4.3)

> Toolbox Main Authors: **Nancy Panousopoulou (FORTH-ICS), Samuel Farrens (CEA), Konstantina Fotiadou (FORTH-ICS), Greg Tsagkatakis (FORTH-ICS)**  
> Year: **2017-2018**   
> Corresponding Author Email: [apanouso@ics.forth.gr](mailto:apanouso@ics.forth.gr)
> Website: [https://github.com/dedale-fet](https://github.com/dedale-fet)  


## Introduction

This repository contains the Python toolbox for distributed sparsity-based learning architectures, along with benchmarking imaging test sets.


The toolbox implements the Dedale Distributed Learning Architecture for solving large-scale imaging problems, associated to:

* Space variant deconvolution of galaxy survey images (package: Distributed Space Variant Deconvolution)
* Hyperspectral and color image super resolution  (package: Distributed Sparce Coupled Dictionary Learning)

Prior referring to the the documentation in each sub-folder on how to use the toolbox for each application, please read the following guidelines for deploying and configuring the toolbox.


## Prerequisities for deploying and using the toolbox

The implementation of the Distributed Learning Architecture considers the use of the [Apache Spark distributed computing framework](https://spark.apache.org/).

### Software packages and Operating System

The prerequisities for installing a Spark-compliant cluster over a set of working terminals are:

* Linux OS (tested with Ubuntu 16.04.3 LTS) 

* SSH client-server packages (e.g., the openssh packages available with the Linux distribution).

* Apache Spark. Tested the version 2.1.1 (pre-build with Apache Hadoop 2.7 and later), which is available for download [here](https://spark.apache.org/downloads.html).

* [Python](https://www.python.org/). Tested with version 2.7.12.

* Java JDK / RE . Tested with SE version 1.8.0


Each of these packages should be installed on all terminals which will comprise the Spark cluster. 

### Spark cluster configuration


1. On each terminal for your cluster extract the prebuild version of Spark into a prefered folder with read/write/execute permissions. In this guide, the preselected folder for all terminals is `$SPARK=/usr/local/spark`

2. On the master node:

	i. Download the folder spark-configurations into a local folder

	ii. Copy contents of spark-configurations into `$SPARK/conf`.

	iii. Define the master host: Edit line 50 of the file `$SPARK/conf/spark-env.sh` to bind the master of cluster to the IP of the master terminal. For example, if the IP of the master terminal is `XXX.XXX.XXX.XXX` then assign:
`SPARK_MASTER_HOST='XXX.XXX.XXX.XXX'`. Save and close the file.

	iv. Define the slave nodes: Open and edit file `$SPARK/conf/slaves` to indicate the IPs of the worker nodes (line 19 and onwards). Save and close the file.

	v. Cluster configuration parameters. The configuration and environmental parameters for the cluster can be tuned at the file `$SPARK/spark-defaults.conf`.

		* Define the port number for the spark cluster web-interface: Edit line 28 of the file `$SPARK/spark-defaults.conf` to indicate the URL for the spark cluster web interface. For example if the IP of the master terminal is `XXX.XXX.XXX.XXX` then assign:

`spark.master spark://XXX.XXX.XXX.XXX:7077` 

		* Define the location of the logging configuration: Edit the value of `-Dlog4j.configuration` at line 34 to indicate the location of the `log4j.properties' file ( `$SPARK\conf\log4j.properties` )

		* (If needed:) Define the memory size allocated at the master for spark calculations by accordingly changing the value of variable `spark.driver.memory` at line 32 (in the current configuration: 8GB of RAM) 

		* (If needed:) Define the memory size allocated at each worker for spark calculations by accordingly changing the value of variable `spark.executor.memory` at line 35 (in the current configuration: 2GB of RAM) 

		* Save and close the `$SPARK/conf/spark-defaults.conf` file. 

Note: For a complete list of tunable parameters for the cluster configuration consult the documentation available [here](https://spark.apache.org/docs/2.1.1/configuration.html)



### Launching/Stopping the cluster

* For starting the cluster: Open a command terminal at the master node and type:

```bash
$ $SPARK/sbin/start-master.sh; $SPARK/sbin/start-slaves.sh 
```
$SPARK is the location of the spark prebuild files (e.g., /usr/local/spark)

* For shutting down the cluster: Open a command terminal at the master node and type:

```bash
$ $SPARK/sbin/stop-master.sh; $SPARK/sbin/stop-slaves.sh 
```

Note: The configuration, launch and stop of the cluster can also be handled through ssh connection at the master node. 



## Reference Documents: 

* K.  Fotiadou, G. Tsagkatakis, P. Tsakalides, "Linear Inverse Problems with Sparsity Constraints," DEDALE DELIVERABLE 3.1, 2016.  

* A. Panousopoulou, S. Farrens, S., Y. Mastorakis, J.L. Starck, P. Tsakalides, "A distributed learning architecture for big imaging problems in astrophysics," In 2017 25th European (pp. 1440-1444). IEEE.

* S.  Farrens,  F.M.  Ngole  Mboula,  and  J.-L.  Starck,  “Space variant deconvolution of galaxy survey images,”  Astronomy and 
& Astrophysics, vol. 601, A66 , 2017.
