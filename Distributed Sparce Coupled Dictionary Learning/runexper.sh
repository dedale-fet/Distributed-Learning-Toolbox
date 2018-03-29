##!/bin/bash

#for i in {0..20}
#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 512 --window 10 --n_iter 100 --partitions 240 --imageN 39542 --bands_h 2187 --bands_l 243 > temp_39542x512_240.txt
#mv log.out log_512x39542_240.out

#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 1024 --window 10 --n_iter 100 --partitions 240 --imageN 39542 --bands_h 2187 --bands_l 243 > temp_39542x1024_240.txt
#mv log.out log_1024x39542_240.out


#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 2056 --window 10 --n_iter 15 --partitions 240 --imageN 39542 --bands_h 2143 --bands_h 243 > temp_39542x2056_240.txt
#mv log.out log_2056x39542_240.out

#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 4092 --window 10 --n_iter 15 --partitions 240 --imageN 39542 --bands_h 2143 --bands_h 243 > temp_39542x4092_240.txt
#mv log.out log_4092x39542_240.out

/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 512 --window 10 --n_iter 100 --partitions 192 --imageN 39542 --bands_h 289 --bands_l 81 --lamba 0.8 > temp_39542x512_192.txt
mv log.out log_512x39542_192.out

/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 1024 --window 10 --n_iter 100 --partitions 192 --imageN 39542 --bands_h 289 --bands_l 81 --lamba 0.8 > temp_39542x1024_192.txt
mv log.out log_1024x39542_192.out


/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 512 --window 10 --n_iter 100 --partitions 140 --imageN 39542 --bands_h 289 --bands_l 81 --lamba 0.8 > temp_39542x512_140.txt
mv log.out log_512x39542_140.out


/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 1024 --window 10 --n_iter 100 --partitions 140 --imageN 39542 --bands_h 289 --bands_l 81 --lamba 0.8 > temp_39542x1024_140.txt
mv log.out log_1024x39542_140.out


/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 512 --window 10 --n_iter 100 --partitions 96 --imageN 39542 --bands_h 289 --bands_l 81 --lamba 0.8 > temp_39542x512_96.txt
mv log.out log_512x39542_96.out


/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 1024 --window 10 --n_iter 100 --partitions 96 --imageN 39542 --bands_h 289 --bands_l 81 --lamba 0.8 > temp_39542x1024_96.txt
mv log.out log_1024x39542_96.out


/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 512 --window 10 --n_iter 100 --partitions 72 --imageN 39542 --bands_h 289 --bands_l 81 --lamba 0.8 > temp_39542x512_72.txt
mv log.out log_512x39542_72.out


/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 1024 --window 10 --n_iter 100 --partitions 72 --imageN 39542 --bands_h 289 --bands_l 81 --lamba 0.8 > temp_39542x1024_72.txt
mv log.out log_1024x39542_72.out

/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 512 --window 10 --n_iter 100 --partitions 48 --imageN 39542 --bands_h 289 --bands_l 81 --lamba 0.8 > temp_39542x512_48.txt
mv log.out log_512x39542_48.out


/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 1024 --window 10 --n_iter 100 --partitions 48 --imageN 39542 --bands_h 289 --bands_l 81 --lamba 0.8 > temp_39542x1024_48.txt
mv log.out log_1024x39542_48.out


/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 2056 --window 10 --n_iter 100 --partitions 48 --imageN 39542 --bands_h 289 --bands_l 81 --lamba 0.8 > temp_39542x2056_48.txt
mv log.out log_2056x39542_48.out

/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 2056 --window 10 --n_iter 100 --partitions 72 --imageN 39542 --bands_h 289 --bands_l 81 --lamba 0.8 > temp_39542x2056_72.txt
mv log.out log_2056x39542_72.out

/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 2056 --window 10 --n_iter 100 --partitions 96 --imageN 39542 --bands_h 289 --bands_l 81 --lamba 0.8 > temp_39542x2056_96.txt
mv log.out log_2056x39542_96.out

/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 2056 --window 10 --n_iter 100 --partitions 140 --imageN 39542 --bands_h 289 --bands_l 81 --lamba 0.8 > temp_39542x2056_140.txt
mv log.out log_2056x39542_140.out

/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip training_np8.py --inputhigh /home/sparkuser/nancy/cdl/cluster/safran/Gray_high_9x17_trs.csv --inputlow /home/sparkuser/nancy/cdl/cluster/safran/Gray_low_9x17_trs.csv --dictsize 2056 --window 10 --n_iter 100 --partitions 192 --imageN 39542 --bands_h 289 --bands_l 81 --lamba 0.8 > temp_39542x2056_192.txt
mv log.out log_2056x39542_192.out


