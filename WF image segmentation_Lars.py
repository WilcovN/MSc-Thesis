# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:34:22 2024

@author: 100551
"""

# import cv2
import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import gc
import numpy as np
# from copy import deepcopy
# from skimage.draw import circle_perimeter
# from scipy.ndimage import gaussian_filter
import time
import os
# import re
from glob import glob
import multiprocessing

from Lars_image_analysis_functions import microscope_FACS_sorting #, tophat, auto_blur_contrast, detect_blobs, select_good_blobs, get_mean_intensity_blob, blobs_to_binary_img, del_overlap_blobs

if __name__ == '__main__':
    filepath = "E:\\Data\\Lars\\LVR019\\2. image analysis_20241219at115441"
    #filepath = '\\\\store.erasmusmc.nl\\department\\gene\\chien_data\\Lab\\Data_and_Analysis\\Lars van Roemburg\\LVR007_co-culture test nr2\\Raw data\\TCC_Lars\\LVR007_whole_dish2_20240412at191427'
    # filepath = '\\\\store.erasmusmc.nl\\department\\gene\\chien_data\\Lab\\Data_and_Analysis\\Lars van Roemburg\\LVR008_stringent wash optimization nr2\\Raw data\\autofocus_20240417at144558'

    try:
        os.mkdir(filepath+"\\analysis plots")
        os.mkdir(filepath+"\\tagging NLS-HaloTag")
        os.mkdir(filepath+"\\tagging mScarlet")
        os.mkdir(filepath+"\\tagging unsure")
    except:
        print("folders already exists")
        
    WL_files = glob(filepath+"\\WhiteLight\\*.tif")
    WL_files.sort()
    
    R_files = glob(filepath+"\\532nm\\*.tif")
    R_files.sort()
    
    N_ROIs = len(R_files)
    
    csv_files = glob(filepath+"\\tagging mScarlet\\*mScar*.csv")
    n_start = len(csv_files)
    
    print("{} ROIs imaged, {} analyzed and {} going to be analyzed now.".format(N_ROIs, n_start, N_ROIs-n_start))
    
    # t = np.zeros(N_ROIs)
    tagging_cells = multiprocessing.Array('i',(N_ROIs, 3))
    
    num_cores = multiprocessing.cpu_count()
    thres_low=500
    thres_high=2000
    bnd_cutoff=1/10.24 #500 pixels, 1/5.12 is 1000 pixels
    
    start = n_start
    ending = N_ROIs
    
    for i in np.arange(start, ending, num_cores): #n_start,N_ROIs,num_cores):
        # print(f"ROI{i:03d}")
        tic = time.perf_counter()
        # file_WL = WL_files[i] #"FOV"+str(i+1)+"_WL_1ms.tif"
        # file_R = R_files[i] #"FOV"+str(i+1)+"_42G_2000ms.tif"
        # print("start")
        
        subprocess = []
        
        for j in range(num_cores):
            if i+j < ending:
                # t = time.perf_counter()
                proc = multiprocessing.Process(target=microscope_FACS_sorting, kwargs={"filepath":filepath, "i":i+j, "file_WL":WL_files[i+j], "file_R":R_files[i+j],"tagging_cells":tagging_cells, "thres_low":thres_low, "thres_high":thres_high, "bnd_ctoff":bnd_cutoff})
                # t2 = time.perf_counter()

                subprocess.append(proc)
                proc.start()
                # t3 = time.perf_counter()
                # print("it takes {} s to create multiproces and {} s to start it".format(t2-t, t3-t2))

                print("ID of process {}: {}".format(j,proc.pid))
                
        for proc in subprocess:
            proc.join()
        
        toc = time.perf_counter()
        # t[i] = toc-tic
        print("{:0.2f} s".format(toc-tic))
        print("{:0.2f}% done".format((i-start+len(subprocess))/(ending-start)*100))
        # print("reading done")
        # blobs_new, yxr_NLS, xyr_NLS, yxr_mScar, xyr_mScar, yxr_unsure, xyr_unsure = microscope_FACS_sorting(img_WL, img_R, thres_low=500, thres_high=3000, bnd_ctoff=1/5.12)
        
    # np.savetxt(filepath+"\\multi summaryCS ROI{:03d}-{:03d}.csv".format(n_start, N_ROIs), tagging_cells, fmt='%d', delimiter=',')
    
    # print("On average it takes {:0.2f} s per ROI".format(np.mean(t[n_start:N_ROIs])))
    # without plotting on average 34.55 s per ROI
    # with plotting on average 38.17 s per ROI
