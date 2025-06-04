# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:47:35 2024

@author: 100551
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter, binary_dilation, binary_closing, binary_erosion, binary_opening
from copy import deepcopy
import gc
import math
from skimage.draw import circle_perimeter
import logging
from scipy.signal import fftconvolve


def auto_blur_contrast(image, minmax=False, thr_perc=0.4, blur=1):
    """
    This function applies an automatic blurring and contrasting to enhance the contrast the most of the input image. 
    Parameters can be changed to fine-tune for own application, or just the min and max intensity value can be used for contrast scaling.

    Parameters
    ----------
    image: (ndarray, dtype=uint)
        input image, can be grayscale or rgb.
    minmax: (bool), optional
        If True, only the minimum and maximum value of the image is used for the contrast enhancement. The default is False.
    thr_perc : float, optional
        the percentage on which the cutoff for contrast enchancement should be. 
        0.1*thr_perc for the lower percentage limit and thr_perc for the upper limit. 
        The default is 0.4 and the range 0 to 10 is recommended. 
    blur : int, optional
        The value for the sigma in the gaussian_filter blurring. The default is 1.

    Returns
    -------
    img : (ndarray, dtype=uint)
        The contrast enhanced and (slightly) blurred output image.
    """

    img = deepcopy(image)

    dt = img.dtype
    m = 255
    if dt != np.uint8:
        if dt == np.uint16:
            m = 65535
        elif dt == np.uint32:
            m = 4294967295
        elif dt == np.uint64:
            m = 18446744073709551616
        else:
            m = 65535
            dt = np.uint16  # if it is not uint yet, we're going to make it uint16 in the end
            #raise TypeError("Only uint images allowed")

    img = gaussian_filter(img, sigma=blur)

    if minmax:
        thr_min = np.min(img)
        thr_max = np.max(img)

    else:
        color = False
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            color = True
        # elif img.dtype != np.uint8:
        #     img = cv2.convertScaleAbs(img, alpha=(255/m))

        histo, bins = np.histogram(img, bins=20000)
        histo = np.cumsum(histo)
        histo = histo / histo[-1] * 100

        min_where = np.where(histo < 0.1*thr_perc)
        max_where = np.where(histo > 100-thr_perc)

        if len(min_where[0]) != 0:
            plc_min = min_where[0][-1]
            thr_min = math.ceil((bins[plc_min] / 2 + bins[plc_min + 1] / 2))
        else:
            #print("no plc in histogram found to be <0.5%")
            thr_min = np.min(image)

        if len(max_where[0]) != 0:
            plc_max = max_where[0][0]
            thr_max = math.floor((bins[plc_max] / 2 + bins[plc_max + 1] / 2))
        else:
            #print("no plc in histogram found to be >99.5%")
            thr_max = np.max(image)

        if color:
            img = gaussian_filter(image, sigma=blur)

    #print("thr_min = {}".format(thr_min))
    #print("thr_max = {}".format(thr_max))
    img[img < thr_min] = thr_min
    img[img > thr_max] = thr_max

    img = (img - thr_min) * np.float64(m/(thr_max-thr_min))
    img[img % 1 < 0.5] = np.floor(img[img % 1 < 0.5])
    img[img % 1 >= 0.5] = np.ceil(img[img % 1 >= 0.5])

    img = np.array(img, dtype=dt)

    return img


def fft_remove_WFbg(image, blur_bg=50):
    """
    Removes the lowest fourier frequencies of an image, 
    and with that removing the background of the image.
    Is not perfect.

    Parameters
    ----------
    image : (ndarray)
        input image, must be one channel
    blur_bg : int, optional
        how big the box should be in the fourier transform image to represent the background/low frequencies. 
        The default is 50.

    Returns
    -------
    im : (ndarray)
        image - background image.
    """

    im = deepcopy(image)

    fft_im = np.fft.fftshift(np.fft.fft2(im))

    fft_im[:int(fft_im.shape[0]/2-blur_bg/2), :] = 0 + 0j
    fft_im[:, :int(fft_im.shape[1]/2-blur_bg/2)] = 0 + 0j
    fft_im[int(fft_im.shape[0]/2+blur_bg/2)+1:, :] = 0 + 0j
    fft_im[:, int(fft_im.shape[1]/2+blur_bg/2)+1:] = 0 + 0j

    ifft_im = np.fft.ifft2(np.fft.ifftshift(fft_im)).real
    im = 1.0 * im - ifft_im

    return im


def detect_blobs(image, min_thres=0.25, max_thres=0.5, min_circ=0.3, max_circ=1.01, min_area=5,
                 max_area=np.inf, min_in=0, max_in=1.01, min_con=0, max_con=1.01, plotting=False):
    """
    Here blob detection from opencv is performed on a binary image (or an image with intensities ranging from 0 to 1).
    You can set the constraints of the blob detection with the parameters of this function.

    If you want more information on the opencv blob detection and how the parameters exactly work:
    https://learnopencv.com/blob-detection-using-opencv-python-c/

    NOTE: In the opencv documentation is stated that the range of most of the parameters are from 0 to 1 (including 1)
    however I experienced that it does not include 1, so I put the upper limit to 1.01 of these parameters to include 1.

    Parameters:
        binary_end_result (ndarray):    The image on which blob detection should be performed.
        min_thres (float):              If the image has a grey scale, the blob detection checks several thresholding
                                        levels beginning with the minimal threshold set by this value. Both min_thres
                                        and max_thres need a specific nonzero value for them to be used.
        max_thres (float):              If the image has a grey scale, the blob detection checks several thresholding
                                        levels ending with the maximal threshold set by this value. Both min_thres
                                        and max_thres need a specific nonzero value for them to be used.
        min_circ (float):               The minimal circularity the blob needs to have to be detected (range:[0,1.01])
        max_circ (float):               The maximal circularity the blob needs to have to be detected (range:[0,1.01])
        min_area (float):               The minimal area in pixels the blob needs to have to be detected (range:[0,inf])
        max_area (float):               The maximal area in pixels the blob needs to have to be detected (range:[0,inf])
        min_in (float):                 The minimal inertia the blob needs to have to be detected (range:[0,1.01])
        max_in (float):                 The maximal inertia the blob needs to have to be detected (range:[0,1.01])
        min_con (float):                The minimal convexity the blob needs to have to be detected (range:[0,1.01])
        max_con (float):                The maximal convexity the blob needs to have to be detected (range:[0,1.01])
        plotting (bool):                If set to True, the detected blobs are shown with red circles in the image.

    Returns:
        key_points (tuple):             The raw output of the opencv blob detection.
        yxr (ndarray):                  Only the y,x position and the radius of the blob in a numpy array.
    """

    # converting the image to a format the opencv blob detection can use
    if image.dtype != np.uint8:
        m = (image.dtype == np.uint16) * 65535 + (image.dtype == np.uint32) * \
            4294967295 + (image.dtype == np.uint64) * 18446744073709551616
        if m == 0:
            raise TypeError("can't convert to uint8 image")
        im = cv2.cvtColor(cv2.convertScaleAbs(
            image, alpha=(255/m)), cv2.IMREAD_GRAYSCALE)
    else:
        im = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)

    # setting the parameters of the blob detections
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.filterByConvexity = False

    # thresholding for grey scale images, both min and max need a specific nonzero value.
    if min_thres and max_thres:
        params.minThreshold = min_thres * 255
        params.maxThreshold = max_thres * 255

    # circularity
    if (min_circ == 0) & (max_circ == 1.01):
        params.filterByCircularity = False
    else:
        params.filterByCircularity = True
        params.minCircularity = min_circ
        params.maxCircularity = max_circ

    # area
    if (min_area == 0) & (max_area == np.inf):
        params.filterByArea = False
    else:
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area

    # inertia
    if (min_in == 0) & (max_in == 1.01):
        params.filterByInertia = False
    else:
        params.filterByInertia = True
        params.minInertiaRatio = min_in
        params.maxInertiaRatio = max_in

    # convexity
    if (min_con == 0) & (max_con == 1.01):
        params.filterByConvexity = False
    else:
        params.filterByConvexity = True
        params.minConvexity = min_con
        params.maxConvexity = max_con

    # creating the detector with the specified parameters
    ver = cv2.__version__.split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # detecting the blobs
    key_points = detector.detect(im)

    # reading out the key_points and putting them in a numpy array
    if len(key_points) != 0:
        yxr = np.zeros((len(key_points), 3), dtype=int)
        for u in range(len(key_points)):
            yxr[u, 1] = int(key_points[u].pt[0])
            yxr[u, 0] = int(key_points[u].pt[1])
            yxr[u, 2] = int(key_points[u].size / 2)  # the size is the diameter
    else:
        yxr = np.array([])

    # Draw them if specified
    if plotting:
        #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 8))
        if len(key_points) != 0:
            for center_y, center_x, radius in zip(yxr[:, 0], yxr[:, 1], yxr[:, 2]):
                circy, circx = circle_perimeter(center_y, center_x, radius,
                                                shape=im.shape)
                circy[circy >= im.shape[0]-1] = im.shape[0] - 2
                circx[circx >= im.shape[1]-1] = im.shape[1] - 2
                circy[circy < 0] = 0
                circx[circx < 0] = 0

                im[circy, circx] = (220, 20, 20, 255)
                im[circy+1, circx+1] = (220, 20, 20, 255)
                im[circy-1, circx+1] = (220, 20, 20, 255)
                im[circy+1, circx-1] = (220, 20, 20, 255)
                im[circy-1, circx-1] = (220, 20, 20, 255)

        # ax.imshow(im)
        # plt.show()

    return np.array(key_points), yxr, im


def tophat(image, filterSize=30):
    """
    Creates the tophat image of the input image. 
    It basically defines the foreground and background in the image.

    Parameters
    ----------
    image : (ndarray)
        input image.
    filterSize : int, optional
        How big the objects are in the foreground. For White light images the recommended value is 30.
        For fluorescence images the recommended value is 120. The default is 30.

    Returns
    -------
    tophat_img : (ndarray)
        tophat output image
    """

    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT, ksize=(filterSize, filterSize))
    tophat_img = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    return tophat_img


def get_mean_intensity_blob(yxr, image, size_factor=2/3):
    """
    For each blob found in the blob detection function, the mean intensity is calculated of the input image. 
    The pixels inlcuded are a square within the circle obtained from the blob detection. 
    The size_factor is how much we use of the area.

    Parameters
    ----------
    yxr : (ndarray)
        The y, x, r coordinates of the blobs detected.
    image : (ndarray)
        The image of which the mean intensities have to be calculated for the blobs
    size_factor : float, optional
        If 1, we take the entire square within the circle. The amount of area used is scaled with this factor.
        The default is 2/3.

    Returns
    -------
    average : (ndarray)
        For each blob the mean intensity in a 1D array.

    """

    average = np.zeros(len(yxr))
    s2 = np.sqrt(2)/size_factor

    for i in range(len(yxr)):
        y, x, r = yxr[i, :]
        y_min = int((y-r/s2) * (y-r/s2 > 0) + 0)
        y_max = int((y+r/s2+1) * (y+r/s2+1 < image.shape[0]) + (
            image.shape[0] - 1) * (y+r/s2+1 >= image.shape[0]))
        x_min = int((x-r/s2) * (x-r/s2 > 0) + 0)
        x_max = int((x+r/s2+1) * (x+r/s2+1 < image.shape[1]) + (
            image.shape[1] - 1) * (x+r/s2+1 >= image.shape[1]))

        average[i] = np.mean(image[y_min:y_max, x_min:x_max])

    return average


def select_good_blobs(blobs, WL_image, tophat_image, factorWL=2, factorTOP=1.8, boundary_cutoff=0):
    """
    To get rid of the blobs created by junk or holes in the image. 
    The function looks at the mean intensities of the blobs in the White Light and tophat image,
    Whitelight image for filtering out junk and tophat image for filtering out holes.
    It sees if any blobs have significant lower intensities which indicates that there is no cell present at that blob.
    The factorWL and factorTOP are the values to define what is an too low mean intensity.

    Parameters
    ----------
    blobs : (tuple)
        The first output of the blob detection function. It contains the cv2.key_points, yxr coordinates and image.
    WL_image : (ndarray)
        The white light image
    tophat_image : (ndarray)
        The tophat image of the white light image.
    factorWL : float, optional
        If the WL intensity of the blob is below the average intensity divided by factorWL, the blob is discarded. The default is 2.
    factorTOP : TYPE, optional
        If the tophat intensity of the blob is below the average intensity divided by factorTOP, the blob is discarded. The default is 1.8.

    Returns
    -------
    blobs_new : (tuple)
        The same as the input blobs but without the discarded blobs.

    """

    average_WL = get_mean_intensity_blob(blobs[1], WL_image)
    average_top = get_mean_intensity_blob(blobs[1], tophat_image)

    mean_average_WL = np.mean(average_WL)
    mean_average_top = np.mean(average_top)

    idx_rm = np.where(average_top < mean_average_top/factorTOP)[0]
    idx_rm = np.append(idx_rm, np.where(
        average_WL < mean_average_WL/factorWL)[0])

    if boundary_cutoff != 0:
        idx_rm = np.append(idx_rm, np.where(
            blobs[1][:, 0] < int(WL_image.shape[0]*boundary_cutoff))[0])
        idx_rm = np.append(idx_rm, np.where(blobs[1][:, 0] > int(
            WL_image.shape[0]*(1-boundary_cutoff)))[0])
        idx_rm = np.append(idx_rm, np.where(
            blobs[1][:, 1] < int(WL_image.shape[1]*boundary_cutoff))[0])
        idx_rm = np.append(idx_rm, np.where(blobs[1][:, 1] > int(
            WL_image.shape[1]*(1-boundary_cutoff)))[0])

    idx_rm = np.unique(idx_rm)

    idx = np.ones(len(blobs[0]), dtype=bool)
    idx[idx_rm] = False

    blobs_new = (blobs[0][idx], blobs[1][idx], blobs[2])

    return blobs_new


def blobs_to_binary_img(yxr, image_shape, circular=False):
    """
    To create a binary image of the blobs. 0 is there is no blob at the pixel, 
    1 if there is a blob at the pixel. You can have a fast option with circular=False and a square is created instead of circle. 
    If circular is True, a real circle is plotted, but it takes significant more time.

    Parameters
    ----------
    yxr : (ndarray)
        the coordinates of the blobs.
    image_shape : (tuple)
        the shape of the image
    circular : (bool), optional
        if a simple fast square or slow circle is used for the blob. The default is False.

    Returns
    -------
    bin_img : (ndarray)
        A binary image of the blobs

    """

    bin_img = np.zeros(image_shape, dtype=bool)

    if circular:
        x_index, y_index = np.meshgrid(
            np.arange(image_shape[1]), np.arange(image_shape[0]))
        for y, x, r in yxr:
            mask = (y_index-y)**2 + (x_index-x)**2 <= r**2
            bin_img[mask] = True
    else:
        for y, x, r, in yxr:
            y_min = int((y-r) * (y-r > 0) + 0)
            y_max = int(
                (y+r+1) * (y+r+1 < image_shape[0]) + (image_shape[0] - 1) * (y+r+1 >= image_shape[0]))
            x_min = int((x-r) * (x-r > 0) + 0)
            x_max = int(
                (x+r+1) * (x+r+1 < image_shape[1]) + (image_shape[1] - 1) * (x+r+1 >= image_shape[1]))

            bin_img[y_min:y_max, x_min:x_max] = True

    return bin_img


def del_overlap_blobs(yxr, bin_img, circular=False, dilation=0):
    """
    This function deletes the blobs that overlap with a contradicting blob. 
    This function can be used in multiple way but the idea is this:
    Blobs without fluorescence will be deleted if they overlap with an unsure or fluorescence blob.
    Blobs with fluorescence will be deleted if they overlap with a blob without fluorescence.
    yxr is the positions of the blobs you want to check and bin_img is the binary image of the blobs you're comparing it to.
    E.g. you put the yxr of the fluorescence blobs and the bin_img of the non-fluorescence blobs to delete the fluorescence blobs that overlap.
    (Note: you will have to do it vice versa to get rid of the non-fluorescent blob)

    Parameters
    ----------
    yxr (ndarray, dtype=int): 
        DESCRIPTION.
    bin_img (ndarray, dtype=bool): 
        DESCRIPTION.
    circular (bool): 
        DESCRIPTION. The default is False.
    dilation (int): 

    Returns
    -------
    yxr_new (ndarray, dtype=int): 

    """

    del_idx = []

    im_shape0 = bin_img.shape[0]
    im_shape1 = bin_img.shape[1]

    if dilation != 0:
        bin_img = binary_dilation(bin_img, structure=np.ones(
            (3, 3), dtype=bool), iterations=dilation)

    if circular:
        x_index, y_index = np.meshgrid(
            np.arange(im_shape1), np.arange(im_shape0))

    for i in range(len(yxr)):
        y, x, r = yxr[i, :]
        if circular:
            mask = (y_index-y)**2 + (x_index-x)**2 <= r**2
            if np.sum(bin_img[mask]) > 0:
                del_idx.append(i)
        else:
            y_min = int((y-r) * (y-r > 0) + 0)
            y_max = int((y+r+1) * (y+r+1 < im_shape0) +
                        (im_shape0 - 1) * (y+r+1 >= im_shape0))
            x_min = int((x-r) * (x-r > 0) + 0)
            x_max = int((x+r+1) * (x+r+1 < im_shape1) +
                        (im_shape1 - 1) * (x+r+1 >= im_shape1))

            if np.sum(bin_img[y_min:y_max, x_min:x_max]) > 0:
                del_idx.append(i)

    yxr_new = np.delete(yxr, del_idx, axis=0)

    return yxr_new


def microscope_FACS_sorting(filepath, i, file_WL, file_R, tagging_cells, min_A=1000, max_A=10000, thres_low=1000, thres_high=3000, bnd_ctoff=1/5.12):
    """


    Parameters
    ----------
    img_WL : TYPE
        DESCRIPTION.
    img_R : TYPE
        DESCRIPTION.
    min_A : TYPE, optional
        DESCRIPTION. The default is 1000.
    max_A : TYPE, optional
        DESCRIPTION. The default is 10000.
    thres_low : TYPE, optional
        DESCRIPTION. The default is 1000.
    thres_high : TYPE, optional
        DESCRIPTION. The default is 3000.

    Returns
    -------
    blobs_new : TYPE
        DESCRIPTION.
    yxr_NLS : TYPE
        DESCRIPTION.
    xyr_NLS : TYPE
        DESCRIPTION.
    yxr_mScar : TYPE
        DESCRIPTION.
    xyr_mScar : TYPE
        DESCRIPTION.
    yxr_unsure : TYPE
        DESCRIPTION.

    """

    img_WL = cv2.imread(file_WL, cv2.IMREAD_UNCHANGED)
    img_R = cv2.imread(file_R, cv2.IMREAD_UNCHANGED)

    # img_R_top = tophat(gaussian_filter(img_R, sigma=5), filterSize=120)
    # img_R_top = auto_blur_contrast(img_R_top, blur=5)

    img_WL_top = tophat(img_WL)
    img_R_top = tophat(gaussian_filter(img_R, sigma=5), filterSize=120)

    img_WL = auto_blur_contrast(img_WL)
    img_WL_top = auto_blur_contrast(img_WL_top)
    #img_WL_wtbg = auto_blur_contrast(img_WL_wtbg)
    img_R = auto_blur_contrast(img_R, blur=5)
    img_R_top = auto_blur_contrast(img_R_top, blur=5)
    #img_R_top_blurred = auto_blur_contrast(img_R_top, blur=10)
    # print("contrasting done")
    #imgthres = cv2.convertScaleAbs(gaussian_filter(img_R_top,sigma=5), alpha=(255/65535))
    #inv_img_WL = np.max(img_WL) - img_WL
    #inv_img_WL_wtbg = np.max(img_WL_wtbg) - img_WL_wtbg
    blobs_WL = detect_blobs(img_WL, min_thres=0.15, max_thres=0.8,
                            min_circ=0.05, min_area=1000, max_area=10000)
    #print("In ROI{}, {} blobs were detected without tophat processing.".format(i,len(blobs_WL[0])))
    blobs_WL_top = detect_blobs(
        img_WL_top, min_thres=0.15, max_thres=0.8, min_circ=0.05, min_area=1000, max_area=10000)
    #print("In ROI{}, {} blobs were detected with tophat processing.".format(i,len(blobs_WL_top[0])))
    # blobs_R = detect_blobs(img_R, min_thres=0.1, max_thres=0.8, min_circ=0.05, min_area=1000, max_area=25000, plotting=True)

    blobs_new = select_good_blobs(
        blobs_WL_top, img_WL, img_WL_top, boundary_cutoff=bnd_ctoff)
    #th3 = cv2.adaptiveThreshold(imgthres,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,201,-3)

    fluor_I = get_mean_intensity_blob(
        blobs_new[1], img_R_top, size_factor=np.sqrt(2))

    idx_mScar = np.where(fluor_I > thres_high)[0]
    idx_NLS = np.where(fluor_I < thres_low)[0]
    idx_co = np.append(idx_mScar, idx_NLS)

    yxr_mScar = blobs_new[1][idx_mScar, :]
    yxr_NLS = blobs_new[1][idx_NLS, :]
    yxr_unsure = np.delete(blobs_new[1], idx_co, axis=0)
       
    bin_img_mScar = blobs_to_binary_img(yxr_mScar, img_WL.shape)
    bin_img_NLS = blobs_to_binary_img(yxr_NLS, img_WL.shape)
    bin_img_unsure = blobs_to_binary_img(yxr_unsure, img_WL.shape)

    yxr_NLS = del_overlap_blobs(yxr_NLS, bin_img_mScar)
    yxr_NLS = del_overlap_blobs(yxr_NLS, bin_img_unsure)
    yxr_mScar = del_overlap_blobs(yxr_mScar, bin_img_NLS)
    yxr_mScar = del_overlap_blobs(yxr_mScar, bin_img_unsure)
    yxr_unsure = del_overlap_blobs(yxr_unsure, bin_img_NLS)
    yxr_unsure = del_overlap_blobs(yxr_unsure, bin_img_mScar)
    
    print("{:04d} is fluor,\t{:04d} is WL,\t{:04d} is unsure".format(
        len(yxr_mScar), len(yxr_NLS), len(yxr_unsure)))

    xyr_NLS = deepcopy(yxr_NLS)
    xyr_NLS[:, [0, 1]] = yxr_NLS[:, [1, 0]]
    xyr_mScar = deepcopy(yxr_mScar)
    xyr_mScar[:, [0, 1]] = yxr_mScar[:, [1, 0]]
    xyr_unsure = deepcopy(yxr_unsure)
    xyr_unsure[:, [0, 1]] = yxr_unsure[:, [1, 0]]

    
    if len(blobs_new[0]) != 0:
        im = deepcopy(blobs_new[2])
        im2 = cv2.cvtColor(cv2.convertScaleAbs(
            img_R_top, alpha=(255/65535)), cv2.IMREAD_GRAYSCALE)
        for ii in range(3):
            if ii == 0:
                yxr = yxr_NLS
                col = (20, 220, 20, 255)
            elif ii == 1:
                yxr = yxr_mScar
                col = (220, 20, 20, 255)
            else:
                yxr = yxr_unsure
                col = (220, 220, 20, 255)

            for center_y, center_x, radius in zip(yxr[:, 0], yxr[:, 1], yxr[:, 2]):
                circy, circx = circle_perimeter(center_y, center_x, radius,
                                                shape=im.shape)
                circy[circy >= im.shape[0]-1] = im.shape[0] - 2
                circx[circx >= im.shape[1]-1] = im.shape[1] - 2
                circy[circy < 0] = 0
                circx[circx < 0] = 0

                im[circy, circx] = col
                im[circy+1, circx+1] = col
                im[circy-1, circx+1] = col
                im[circy+1, circx-1] = col
                im[circy-1, circx-1] = col

                im2[circy, circx] = col
                im2[circy+1, circx+1] = col
                im2[circy-1, circx+1] = col
                im2[circy+1, circx-1] = col
                im2[circy-1, circx-1] = col

    # tagging_cells[int(i),0] = int(xyr_mScar.shape[0])
    # tagging_cells[int(i),1] = int(xyr_NLS.shape[0])
    # tagging_cells[int(i),2] = int(xyr_unsure.shape[0])

    fig, ax = plt.subplots(ncols=2, figsize=(40, 40))
    ax[0].imshow(im)
    ax[0].set_title(
        "ROI{:03d} mScarlet-red: {}, NLS-HaloTag-green: {}, unsure-yellow: {}".format(i, xyr_mScar.shape[0], xyr_NLS.shape[0], xyr_unsure.shape[0]))
    ax[1].imshow(im2)
    ax[1].set_title(
        "ROI{:03d} mScarlet-red: {}, NLS-HaloTag-green: {}, unsure-yellow: {}".format(i, xyr_mScar.shape[0], xyr_NLS.shape[0], xyr_unsure.shape[0]))
    # fig.savefig(filepath+"\\analysis plots\\ROI{:03d}.png".format(i))
    plt.show()
    gc.collect()
    
    # yxr_mScar[:,2] = fluor_I[idx_mScar]
    # yxr_NLS[:,2] = fluor_I[idx_NLS]
    # yxr_unsure[:,2] = np.delete(fluor_I, idx_co, axis=0)
    
    # np.savetxt(filepath +
    #            f"\\tagging NLS-HaloTag\\Lars_NLS_ROI{i:02d}_Galvo_CS{xyr_NLS.shape[0]:05d}.csv", xyr_NLS, fmt='%d', delimiter=',')
    # np.savetxt(filepath +
    #            f"\\tagging mScarlet\\Lars_mScar_ROI{i:02d}_Galvo_CS{xyr_mScar.shape[0]:05d}.csv", xyr_mScar, fmt='%d', delimiter=',')
    # np.savetxt(
    #     filepath+f"\\tagging unsure\\Lars_unsure_ROI{i:02d}_Galvo_CS{xyr_unsure.shape[0]:05d}.csv", xyr_unsure, fmt='%d', delimiter=',')

    
    return blobs_new, yxr_NLS, xyr_NLS, yxr_mScar, xyr_mScar, yxr_unsure, xyr_unsure, fluor_I, idx_NLS, idx_mScar, idx_co


def find_shift(img1, img2, y_pos=4/10, x_pos=4/10, y_size=1/10, x_size=1/10):
    """
    Shift the img_after in x and y position to have the best overlay with img_before.
    This is done with finding the maximum in a convolution.

    Parameters:
        img_before (ndarray):   The image which will be shifted to for the best overlay.
        img_after (ndarray):    The image which will shift to img_before for the best overlay.
        max_shift (int):        It is assumed that the images are roughly at the same position,
                                so no large shifts are needed. On all sides of the convolution image,
                                1/max_shift of the image is set to be zero. So with 1/8 you have a max shift of 3/8 in
                                x and y direction (-3/8*img_shape, 3/8*img_shape). If you don't want a constraint on
                                the shift, set max_shift to <=1.

    Returns:
        img_after (ndarray):    The shifted image, the values outside the boundaries of the image are set to the max
                                value of the image so that in further processing steps those regions are not used.
        shift (ndarray):        The shift values in dy, dx
    """

    # max shift is between 1 and inf, the higher, the higher the shift can be
    # if it is 1 no constraints will be given on the shift
    img_before = deepcopy(img1)
    img_after = deepcopy(img2)

    mean_before = np.mean(img_before)
    mean_after = np.mean(img_after)

    img_before = img_before - mean_before
    img_after = img_after - mean_after

    y0 = int(img_before.shape[0] * y_pos)
    y1 = int(img_before.shape[0] * (y_pos+y_size))
    x0 = int(img_before.shape[1] * x_pos)
    x1 = int(img_before.shape[1] * (x_pos+x_size))

    conv = fftconvolve(img_before[y0:y1, x0:x1],
                       img_after[y0:y1, x0:x1][::-1, ::-1])

    # plt.imshow(conv)
    # plt.show()

    # max_shift constraints
    # if (int(conv.shape[0] / 4 + (y1-y0) / max_shift) > 0) & (max_shift > 1):
    #     m = np.min(conv) - 1
    #     conv[:int(conv.shape[0] / 4 + (y1-y0) / max_shift), :] = m
    #     conv[-int(conv.shape[0] / 4 + (y1-y0) / max_shift):, :] = m
    #     conv[:, :int(conv.shape[1] / 4 + (x1-x0) / max_shift)] = m
    #     conv[:, -int(conv.shape[1] / 4 + (x1-x0) / max_shift):] = m

    # plt.imshow(conv)
    # plt.show()

    # calculating the shift in y and x with finding the maximum in the convolution
    if np.max(conv) > 0:
        shift = np.where(conv == np.max(conv))
        shift = np.asarray(shift)
        shift[0] = shift[0] - (conv.shape[0] - 1) / 2
        shift[1] = shift[1] - (conv.shape[1] - 1) / 2
    else:
        logging.warning(
            "In overlay(): No maximum was found in the convolution. No shift is applied.")
        shift = np.array([[0], [0]])

    img_before = img_before + mean_before
    img_after = img_after + mean_after

    dy_pix = int(shift[0][0])
    dx_pix = int(shift[1][0])

    logging.debug("dx is {} pixels".format(dx_pix))
    logging.debug("dy is {} pixels".format(dy_pix))

    return shift


def shift_image(img, dy_pix, dx_pix):
    img_after = deepcopy(img)

    # shifting the after milling image towards the before milling image to overlap nicely
    if dx_pix > 0:
        img_after[:, dx_pix:] = img_after[:, :-dx_pix]
        img_after[:, :dx_pix] = np.max(img_after)
        # img_before = img_before[:, dx_pix:]
        # img_after = img_after[:, dx_pix:]
    elif dx_pix < 0:
        img_after[:, :dx_pix] = img_after[:, -dx_pix:]
        img_after[:, dx_pix:] = np.max(img_after)
        # img_before = img_before[:, :dx_pix]
        # img_after = img_after[:, :dx_pix]
    if dy_pix > 0:
        img_after[dy_pix:, :] = img_after[:-dy_pix, :]
        img_after[:dy_pix, :] = np.max(img_after)
        # img_before = img_before[dy_pix:, :]
        # img_after = img_after[dy_pix:, :]
    elif dy_pix < 0:
        img_after[:dy_pix, :] = img_after[-dy_pix:, :]
        img_after[dy_pix:, :] = np.max(img_after)
        # img_before = img_before[:dy_pix, :]
        # img_after = img_after[:dy_pix, :]

    return img_after

def meta_find_shift(img1, img2, size=1/10, start_num=2, max_dist=20):
    shift_found = False
    tries = 0
    num_shifts = start_num

    while not shift_found:
        if tries > 5:
            print("After 5 tries, did still not work")
            shifty = 0
            shiftx = 0
            break

        shifts = np.zeros((num_shifts, 2))
        posy = np.zeros(num_shifts)
        posx = np.zeros(num_shifts)
        # idea: break up low 0 to high 1- size in num_shifts parts and for each choose one random number, in this way they'll never overlap
        for j in range(num_shifts):
            posy[j] = np.random.uniform(low=j*(1-size)/num_shifts, high=(j+1)*(1-size)/num_shifts)
            posx[j] = np.random.uniform(low=j*(1-size)/num_shifts, high=(j+1)*(1-size)/num_shifts)            
        np.random.shuffle(posy)
        np.random.shuffle(posx)
        
        for j in range(num_shifts):
            shift = find_shift(
                img1, img2, y_pos=posy[j], x_pos=posx[j], y_size=size, x_size=size)
            shifts[j, 0] = shift[0][0] # y
            shifts[j, 1] = shift[1][0] # x

        dif_dy = np.zeros((len(shifts), len(shifts)), dtype=int) + int(img1.shape[0])
        dif_dx = np.zeros((len(shifts), len(shifts)), dtype=int) + int(img1.shape[1])

        for ii in np.arange(0, len(shifts), 1):
            for jj in np.arange(ii+1, len(shifts), 1):
                dif_dy[ii, jj] = np.abs(shifts[ii, 0] - shifts[jj, 0])
                dif_dx[ii, jj] = np.abs(shifts[ii, 1] - shifts[jj, 1])
        
        close_y = dif_dy <= max_dist
        close_x = dif_dx <= max_dist
        
        close = close_y * close_x
        
        if np.sum(close) > 0:
            idx = np.where(close)
            
            if np.sum(close) > 1:
                idx_1 = idx[0]
                idx_2 = idx[1]
                
                mean_dy = np.zeros(len(idx_1))
                mean_dx = np.zeros(len(idx_1))
                
                for j in range(np.sum(close)):
                    mean_dy[j] = shifts[idx_1[j],0] / 2 + shifts[idx_2[j],0] / 2
                    mean_dx[j] = shifts[idx_1[j],1] / 2 + shifts[idx_2[j],1] / 2
                # now here just the whole average is taken even if some combination are potentially wrong, not sure if this will be ever the case tho
            else:
                idx_1 = idx[0][0]
                idx_2 = idx[1][0]
                
                mean_dy = shifts[idx_1,0] / 2 + shifts[idx_2,0] / 2
                mean_dx = shifts[idx_1,1] / 2 + shifts[idx_2,1] / 2
            
            shifty = int(np.mean(mean_dy))
            shiftx = int(np.mean(mean_dx))
            if shifty != 0 or shiftx != 0:
                shift_found = True
                print("It took {} tries to find the shift".format(tries+1))
            
        else:
            tries+=1
            num_shifts+=1
            size+=1/10
    
    return shifty, shiftx

def load_images_and_find_shift(file_path, i, file1, file2, coord, dz, endsize=81, max_dist=10, min_shift=30):
    """


    Parameters
    ----------
    file_path : TYPE
        DESCRIPTION.
    file1 : TYPE
        DESCRIPTION.
    file2 : TYPE
        DESCRIPTION.
    i : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    img1_after : TYPE
        DESCRIPTION.
    img2_after : TYPE
        DESCRIPTION.

    """

    # print(i)
    img1 = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)

    img1 = gaussian_filter(tophat(img1), sigma=1)
    img2 = gaussian_filter(tophat(img2), sigma=1)

    s1y, s1x = meta_find_shift(img1, img2, max_dist=max_dist)
    s2y, s2x = meta_find_shift(img1, img2, max_dist=max_dist)
    # roughly 1 in 10.000 will go wrong in this way statistically as with 1 meta_find_shift I saw 1 in 100 went wrong so with two it is 1 in 100*100
    
    if np.abs(s1y-s2y) <= max_dist and np.abs(s1x-s2x) <= max_dist and np.abs(s1y) > min_shift and np.abs(s2y) > min_shift and np.abs(s1x) > min_shift and np.abs(s2x) > min_shift:
        shifty = int(s1y/2+s2y/2)
        shiftx = int(s1x/2+s2x/2)
    else:
        print("first meta try did not work, trying again")
        tries = 0
        while np.abs(s1y-s2y) > max_dist or np.abs(s1x-s2x) > max_dist or np.abs(s1y) < min_shift or np.abs(s2y) < min_shift or np.abs(s1x) < min_shift or np.abs(s2x) < min_shift:
            if tries > 2:
                logging.warning("AFTER A LOT OF TRIES, IT FAILED MISERABLY")
                break
            s1y, s1x = meta_find_shift(img1, img2, max_dist=max_dist)
            s2y, s2x = meta_find_shift(img1, img2, max_dist=max_dist)
            
            if tries > 1:
                s1y, s1x = meta_find_shift(img1, img2, max_dist=max_dist, size=1)
                #s2y, s2x = meta_find_shift(img1, img2, max_dist=max_dist, size=1)
                s2y = s1y # as with size 1 they will be the same
                s2x = s1x
            
            tries+=1
        print("It took {} meta tries".format(tries))
        shifty = int(s1y/2+s2y/2)
        shiftx = int(s1x/2+s2x/2)
            
    print("dy: {}".format(shifty))
    print("dx: {}".format(shiftx))

    shift = np.zeros(2)
    shift[0] = shifty
    shift[1] = shiftx
    
    coord_new = coord - np.array([3.1295 * shifty, 3.1295 * shiftx, dz, 0, 0], dtype=int)
    
    temp_ROI = np.zeros((endsize, 5), dtype=int)
    temp_ROI[i, :] = coord_new
    
    np.savetxt(file_path+f"\\corrected ROI files\\tempROI{i:03d}.txt", temp_ROI, fmt='%d', delimiter='\t')
    
    img2_after = shift_image(img2, shifty, shiftx)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    ax[0, 0].imshow(auto_blur_contrast(img1[3000:3500, 3000:3500]))
    # ax[0,0].set_title("0,0")
    ax[1, 0].imshow(auto_blur_contrast(img2[3000:3500, 3000:3500]))
    # ax[1,0].set_title("1,0")
    ax[0, 1].imshow(auto_blur_contrast(img1[3000:3500, 3000:3500]))
    # ax[0,1].set_title("0,1")
    ax[1, 1].imshow(auto_blur_contrast(img2_after[3000:3500, 3000:3500]))
    # ax[1,1].set_title("1,1")
    fig.savefig(file_path+"\\shift plots\\ROI{:03d}.png".format(i))
    
    # NLS = np.loadtxt(NLS_csv, delimiter=",", dtype=int)
    # mScar = np.loadtxt(mScar_csv, delimiter=",", dtype=int)
    # unsure = np.loadtxt(unsure_csv, delimiter=",", dtype=int)
    # if len(NLS) > 0:
    #     NLS += np.array([-shiftx, -shifty, 100])
    # if len(mScar) > 0:
    #     mScar += np.array([-shiftx, -shifty, 100])
    # if len(unsure) > 0:
    #     unsure += np.array([-shiftx, -shifty, 100])

    
    # np.savetxt(file_path +
    #            f"\\corrected tagging NLS-HaloTag\\Lars_NLS_ROI{i:02d}_Galvo_CS{NLS.shape[0]:05d}.csv", NLS, fmt='%d', delimiter=',')
    # np.savetxt(file_path +
    #            f"\\corrected tagging mScarlet\\Lars_mScar_ROI{i:02d}_Galvo_CS{mScar.shape[0]:05d}.csv", mScar, fmt='%d', delimiter=',')
    # np.savetxt(
    #     file_path+f"\\corrected tagging unsure\\Lars_unsure_ROI{i:02d}_Galvo_CS{unsure.shape[0]:05d}.csv", unsure, fmt='%d', delimiter=',')

    gc.collect()

    return shift  # , img1_after, img2_after
