# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:52:41 2024

@author: 100551
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter, binary_dilation, binary_closing, binary_erosion, binary_opening
from copy import deepcopy
import gc
# import math
#from skimage.draw import circle_perimeter
# import time
import os
# import re
from glob import glob

def downsize_image(image, factor_y=10, factor_x=10, img_type=np.uint8):
    """
    Here we downsize the image from a nxm to an n/f x m/f size with f being the downscale factor. If n % f != 0 and/or
    m % f != 0, the remainder pixels of the nxm image are not used in the downsizing to the n/f x m/f image.

    Parameters:
        image (ndarray):    The input image.
        factor (float):     The factor of downscaling.
        img_type (numpy):   The image type you want it in (not what it is).
                            
    Returns:
        new_img (ndarray):  The downscaled image.

    """
    if image.dtype == img_type:
        temp_img = np.zeros((int(image.shape[0] / factor_y), image.shape[1]), dtype=img_type)
        new_img = np.zeros((int(image.shape[0] / factor_y), int(image.shape[1] / factor_x)), dtype=img_type)

        for e in range(temp_img.shape[0]):
            temp_img[e, :] = np.mean(image[int(factor_y * e):int(factor_y * e + factor_y), :], axis=0)
        for j in range(new_img.shape[1]):
            new_img[:, j] = np.mean(temp_img[:, int(factor_x * j):int(factor_x * j + factor_x)], axis=1)

        return new_img
    
    elif image.dtype == np.uint16 and img_type == np.uint8:
        image = cv2.convertScaleAbs(image, alpha=(255/65535))

        temp_img = np.zeros((int(image.shape[0] / factor_y), image.shape[1]), dtype=img_type)
        new_img = np.zeros((int(image.shape[0] / factor_y), int(image.shape[1] / factor_x)), dtype=img_type)

        for e in range(temp_img.shape[0]):
            temp_img[e, :] = np.mean(image[int(factor_y * e):int(factor_y * e + factor_y), :], axis=0)
        for j in range(new_img.shape[1]):
            new_img[:, j] = np.mean(temp_img[:, int(factor_x * j):int(factor_x * j + factor_x)], axis=1)
    
        return new_img
    else:
        raise TypeError("downsize image() only works with 8 and 16 bit images, please convert it to one of the two first or contact Lars to ask to add your type to the function.")

ds_factor = 10 # image should be perfectly divided in x and y by this factor, if not, errors will occur probably
filepath = r"\\store\department\gene\chien_data\Lab\Data_and_Analysis\Wilco van Nes\WvN016_546_strategy2_20250308\test_run_5x5_20250308at095255"
# filepath = "\\\\store.erasmusmc.nl\\department\\gene\\chien_data\\Lab\\Data_and_Analysis\\Lars van Roemburg\\LVR018\\LVR018\\0. whole dish scan_20241127at111411"
txt = np.loadtxt(filepath+os.sep+"ROIs5x5.txt", dtype=int)

x_pos = np.unique(np.array(np.array(txt[:,1])/10000,dtype=int))
y_pos = np.unique(np.array(np.array(txt[:,0])/10000,dtype=int))

idx_x = np.zeros(len(txt), dtype=int)
idx_y = np.zeros(len(txt), dtype=int)

N_cols = len(x_pos)
N_rows = len(y_pos)
WL_files = glob(filepath+"\\WhiteLight\\*.tif")
files_idx = np.zeros(len(WL_files),dtype=int)
pos_ROIi = len(min(WL_files,key=len))-5

for i in range(len(WL_files)):
    files_idx[i] = int(WL_files[i][pos_ROIi:-4])

WL_files_sorted = []
for i in range(len(WL_files)):
    WL_files_sorted.append(WL_files[np.where(files_idx==i)[0][0]])
    
BIG_img = np.zeros((int(N_rows*5120/ds_factor), int(N_cols*5120/ds_factor)), dtype=np.uint8) # we now assume all the images we use are 5120x5120 pixels


for i in range(len(txt)):
    img = cv2.imread(WL_files_sorted[i], cv2.IMREAD_UNCHANGED)
    # img = cv2.convertScaleAbs(img, alpha=(255/65535))
    img = downsize_image(img, factor_y=ds_factor,factor_x=ds_factor, img_type=np.uint8)
    
    x = txt[i,1]
    y = txt[i,0]
    
    idx = np.where(x_pos==int(x/10000))[0][0]
    idy = np.where(y_pos==int(y/10000))[0][0]
    idx_x[i] = idx
    idx_y[i] = idy
    
    # plt.imshow(BIG_img)
    # plt.axis('off')
    # plt.show()
    
    BIG_img[int(idy*5120/ds_factor):int((idy+1)*5120/ds_factor), int(idx*5120/ds_factor):int((idx+1)*5120/ds_factor)] = img

    print("{:0.2f}% done".format((i+1)/len(txt)*100))
    gc.collect()

plt.imshow(BIG_img)
plt.axis('off')
plt.show()
cv2.imwrite(filepath+os.sep+"WL_frame0_stitched.tif", BIG_img)


# iter_closing = int(3000/ds_factor)
# # possiblity to explore: 
# #bin_img_extended = np.zeros((bin_img.shape[0]+2*iter_closing, bin_img.shape[1]+2*iter_closing), dtype=np.uint8)
# #bin_img_extended[iter_closing:-iter_closing, iter_closing:-iter_closing] = BIG_img > 0.7 * np.max(BIG_img)

# # how it goes now
# bin_img = BIG_img > 1 * np.max(BIG_img)

# bin_img2 = binary_opening(bin_img, iterations=int(iter_closing/60))
# bin_img2 = binary_closing(bin_img2, iterations=iter_closing, structure=np.ones((3,3),dtype=bool))
# bin_img2 = binary_opening(bin_img2, iterations=int(iter_closing/2), structure=np.ones((3,3),dtype=bool))

# bin_img2[:iter_closing, :] = bin_img2[iter_closing:2*iter_closing,:]
# bin_img2[:, :iter_closing] = bin_img2[:, iter_closing:2*iter_closing]
# bin_img2[-iter_closing:, :] = bin_img2[-2*iter_closing:-iter_closing, :]
# bin_img2[:, -iter_closing:] = bin_img2[:, -2*iter_closing:-iter_closing]

# #bin_img2 = binary_dilation(bin_img2, iterations=20, structure=np.ones((3,3),dtype=bool))

# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(bin_img)
# ax[0].axis('off')
# ax[1].imshow(bin_img2)
# ax[1].axis("off")
# plt.show()

# boundary_ROIs = []
# dish_ROIs = []
# rm_i = []
# rm_i = np.append([3,4,8,12,13,17,21,22,29,30,34], np.arange(38,66))
# rm_i = np.append(rm_i, [69,73,74,81,82,86,90,91,95,99,100])
# rm_i = np.append(rm_i, np.arange(104,130))
# rm_i = np.append(rm_i, [133,134,138,142,143,147,151,152,159,160,164])
# rm_i = np.append(rm_i, np.arange(168, 182)) #[12,13,38,39,64,65,90,91,116,117,142,143,168] #np.append([12,13,38,39,64,65,90,91,116,117,142,143,168], np.arange(169,182))#np.append(4+13*np.arange(13), 12+13*np.arange(13))
    
# for i, idy, idx in zip(range(len(txt)), idx_y, idx_x):
#     result = np.sum(bin_img2[int(idy*5120/ds_factor):int((idy+1)*5120/ds_factor), int(idx*5120/ds_factor):int((idx+1)*5120/ds_factor)])
#     if result > 0.1 * 5120/ds_factor * 5120/ds_factor:
#         boundary_ROIs.append(i)
#         print("ROI{:03d}\tindex {}, {}\t\tBOUNDARY".format(i, idy, idx))
#     else:
#         if i not in rm_i:
#             dish_ROIs.append(i)
#             print("ROI{:03d}\tinex {}, {}\t\tDISH".format(i, idy, idx))
#         else:
#             boundary_ROIs.append(i)
#             print("ROI{:03d}\tindex {}, {}\t\tBOUNDARY".format(i, idy, idx))

# end_img = deepcopy(BIG_img)

# for i in boundary_ROIs:
#     end_img[int(idx_y[i]*5120/ds_factor):int((idx_y[i]+1)*5120/ds_factor), int(idx_x[i]*5120/ds_factor):int((idx_x[i]+1)*5120/ds_factor)] = 0

# plt.imshow(end_img)
# plt.axis("off")
# plt.show()

# dish_ROIs_txt = txt[dish_ROIs, :]

# center_ROIs = [10,13,16,37,40,43,64,67,70] # it is important this is the same as for center_focus_to_all.py because the order of the text file matters!

# center_ROIs_txt = np.zeros((9,7), dtype=int)
# center_ROIs_txt[:, :5] = np.array(txt[dish_ROIs, :][center_ROIs, :], dtype=int)
# center_ROIs_txt[:, 5:] = np.array([1, 1000])

# np.savetxt(filepath+os.sep+"ROI_dish.txt", np.array(txt[dish_ROIs, :],dtype=int), fmt='%d', delimiter='\t')
# np.savetxt(filepath+os.sep+"ROIs_for_focusing.txt", center_ROIs_txt, fmt='%d', delimiter='\t')

