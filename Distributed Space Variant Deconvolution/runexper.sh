##!/bin/bash
#for i in {0..20}
#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip,sf_tools.zip  sf_deconvolve.py -i dedale_data/cube_10000_obj_var_euclid_sig0.01.npy -p dedale_data/euclid_psf_wl0.6.npy_10000_random_PSFs.npy --mode lowr --pn 72 --n_iter 150 > temp_lowr_72_10000_0p01.txt
#mv log.out log_lowr_72_10000_0p01.out
#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip,sf_tools.zip  sf_deconvolve.py -i dedale_data/cube_10000_obj_var_euclid_sig0.01.npy -p dedale_data/euclid_psf_wl0.6.npy_10000_random_PSFs.npy  --mode lowr --pn 48 --n_iter 150 > temp_lowr_48_10000_0p01.txt
#mv log.out log_lowr_48_10000_0p01.out
#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip,sf_tools.zip  sf_deconvolve.py -i dedale_data/cube_10000_obj_var_euclid_sig0.01.npy -p dedale_data/euclid_psf_wl0.6.npy_10000_random_PSFs.npy --mode lowr --pn 96 --n_iter 150 > temp_lowr_96_10000_0p01.txt
#mv log.out log_lowr_96_10000_0p01.out
/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip,sf_tools.zip  sf_deconvolve.py -i dedale_data/cube_10000_obj_var_euclid_sig0.01.npy -p dedale_data/euclid_psf_wl0.6.npy_10000_random_PSFs.npy --mode lowr --pn 140 --n_iter 150 > temp_lowr_140_10000_0p01.txt
mv log.out log_lowr_140_10000_0p01.out
#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip,sf_tools.zip  sf_deconvolve.py -i dedale_data/cube_10000_obj_var_euclid_sig0.01.npy -p dedale_data/euclid_psf_wl0.6.npy_10000_random_PSFs.npy --mode sparse --pn 48 > temp_sparse_48_10000_0p01.txt
#mv log.out log_sparse_48_10000_0p01.out
#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip,sf_tools.zip  sf_deconvolve.py -i dedale_data/cube_10000_obj_var_euclid_sig0.01.npy -p dedale_data/euclid_psf_wl0.6.npy_10000_random_PSFs.npy  --mode sparse --pn 72 > temp_sparse_72_10000_0p01.txt
#mv log.out log_sparse_72_10000_0p01.out
#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip,sf_tools.zip  sf_deconvolve.py -i dedale_data/cube_10000_obj_var_euclid_sig0.01.npy -p dedale_data/euclid_psf_wl0.6.npy_10000_random_PSFs.npy --mode sparse --n_iter 300 --pn 96 > temp_sparse_96_10000_0p01_n300.txt
#mv log.out log_sparse_96_10000_0p01_n300.out
#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip,sf_tools.zip  sf_deconvolve.py -i dedale_data/cube_10000_obj_var_euclid_sig0.01.npy -p dedale_data/euclid_psf_wl0.6.npy_10000_random_PSFs.npy --mode sparse --pn 140 > temp_sparse_140_10000_0p01.txt
#mv log.out log_sparse_140_10000_0p01.out
#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip,sf_tools.zip  sf_deconvolve.py -i dedale_data/high_snr_sample.npy -p dedale_data/psf_20000.npy --mode lowr --pn 48 > temp_lowr_48_20000_hSNR.txt
#mv log.out log_lowr_48_20000_hSNR.out
#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip,sf_tools.zip  sf_deconvolve.py -i dedale_data/high_snr_sample.npy -p dedale_data/psf_20000.npy  --mode lowr --pn 72 > temp_lowr_72_20000_hSNR.txt
#mv log.out log_lowr_72_20000_hSNR.out
#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip,sf_tools.zip  sf_deconvolve.py -i dedale_data/high_snr_sample.npy -p dedale_data/psf_20000.npy --mode lowr --n_iter 300 --pn 96 > temp_lowr_96_20000_hSNR_300_n300.txt
#mv log.out log_lowr_96_20000_hSNR_n300.out
#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip,sf_tools.zip  sf_deconvolve.py -i dedale_data/high_snr_sample.npy -p dedale_data/psf_20000.npy --mode lowr --pn 140 --n_iter 150 > temp_lowr_140_20000_hSNR.txt
#mv log.out log_lowr_140_20000_hSNR.out
#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip,sf_tools.zip  sf_deconvolve.py -i dedale_data/high_snr_sample.npy -p dedale_data/psf_20000.npy --mode lowr --pn 48 --n_iter 150 > temp_lowr_48_20000_hSNR.txt
#mv log.out log_lowr_48_20000_hSNR.out
#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip,sf_tools.zip  sf_deconvolve.py -i dedale_data/mid_snr_sample.npy -p dedale_data/psf_20000.npy  --mode sparse --n_iter 300 --pn 96 > temp_sparse_72_20000_n300_mSNR.txt
#mv log.out log_sparse_96_20000_mSNR_n300.out
#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip,sf_tools.zip  sf_deconvolve.py -i dedale_data/low_snr_sample.npy -p dedale_data/psf_20000.npy --mode lowr --n_iter 150 --pn 96 > temp_lowr_96_20000_lSNR.txt
#mv log.out log_lowr_96_20000_lSNR.out
#/usr/local/TensorFlowOnSpark/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://147.52.17.68:7077 --py-files lib.zip,sf_tools.zip  sf_deconvolve.py -i dedale_data/high_snr_sample.npy -p dedale_data/psf_20000.npy --mode lowr --n_iter 150 --pn 96 > temp_lowr_96_20000_hSNR.txt
#mv log.out log_lowr_96_20000_hSNR_n300.out
