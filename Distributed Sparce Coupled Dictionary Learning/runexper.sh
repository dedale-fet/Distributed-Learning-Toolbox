$SPARK/bin/spark-submit --master spark://<IP of master node>:7077 --py-files lib.zip Distributed_SCDL.py --inputhigh <input samples at high resolution>.csv --inputlow <input samples at low resolution>.csv --dictsize <dictionary size> --n_iter <number of optimization iterations> --partitions <number of blocks per RDD>  --imageN <number of training samples>  --bands_h <high resolution dimensions> --bands_l <low resolution dimensions> > <application log file>.txt

mv log.out <spark log file>.out
