$SPARK/bin/spark-submit --master spark://<IP of master node>:7077 --py-files lib.zip,sf_tools.zip  dl_psf_deconvolve.py -i <input stack of noisy data>.npy -p <input psf>.npy --mode <optimization mode> --pn <number of blocks per RDD> --n_iter <number of optimization iterations> > <application log file>.txt
mv log.out <spark log file>.out
