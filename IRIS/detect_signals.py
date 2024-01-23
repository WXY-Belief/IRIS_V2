#!/usr/bin/env python3
"""
This class is used to detect fluorescence signal in each channel.

Usually, chemical reaction or taking photo in local region will trigger the generating of different quality
fluorescence signal in a image, like low gray scale or indistinction between fluorescence signal and background.

A Morphological transformation is called to expose a majority of high quality fluorescence signal as blobs. In this
process, large scale transformator is used to expose dissociative fluorescence signal and small one is used to treat
the adjacent signal as accurately as possible.

When fluorescence signals are exposed, a simple blob detection algorithm is called for blob locating. In our
practice, not only dense fluorescence signal but also sparse blob can be detected by this parameter-optimized
algorithm, while the ambiguous ones will be abandoned. After detection, for each detected blobs, blob's base
score, which is calculated by their gray scale in core (4x4) region being subtracted by surrounding (10x10), is
recorded to calculate base quality in the next step.
"""
import os
import numpy as np
import cv2
from cv2 import (getStructuringElement, morphologyEx, GaussianBlur, convertScaleAbs,
                 SimpleBlobDetector, SimpleBlobDetector_Params,
                 MORPH_ELLIPSE, MORPH_TOPHAT)
######################
# Alternative option #
######################
# from cv2 import (Laplacian,
#                  CV_32F)
#####################
from numpy import (array, zeros, ones, reshape, sum, divide, floor, around, abs, max, int, float32, uint8, bool_, fft)
from scipy.stats import mode

from .call_bases import (image_model_pooling_Ke, pool2base)


def channel_point(point, img, save_path):
    img = np.zeros(img.shape)
    for item in point:
        cv2.circle(img, (int(item[1]), int(item[0])), radius=3, color=(255, 255, 255), thickness=-1)
    cv2.imwrite(os.path.join(save_path), img)


def __hpf(f_img):
    """
    High-pass Filter

    :param f_img: Input image
    :return: Filtered image
    """
    row, col = f_img.shape

    masker_window = ones((row, col), dtype=bool_)
    masker_window[int(row / 2) - int(row * 0.2):int(row / 2) + int(row * 0.2),
    int(col / 2) - int(col * 0.2):int(col / 2) + int(col * 0.2)] = 0

    f_img = abs(fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(f_img)) * masker_window)))
    f_img = convertScaleAbs(f_img / max(f_img) * 255)

    return f_img


def detect_blobs_Ke(f_cycle, cycle_path, temp_flag, blob_params_detect):
    """
    For detect the fluorescence signal.

    Input registered image from different channels.
    Returning the grey scale model.

    :param f_cycle: A image matrix in the 3D common data tensor.
    :return: A base box of this cycle, which store their coordinates, base and its error rate.
    """
    channel_A = f_cycle[0]
    channel_T = f_cycle[1]
    channel_C = f_cycle[2]
    channel_G = f_cycle[3]

    greyscale_model_A = zeros(channel_A.shape, dtype=float32)
    greyscale_model_T = zeros(channel_T.shape, dtype=float32)
    greyscale_model_C = zeros(channel_C.shape, dtype=float32)
    greyscale_model_G = zeros(channel_G.shape, dtype=float32)

    ###############################################################################
    # Here, a morphological transformation, Tophat, under a 15x15 ELLIPSE kernel, #
    # is used to expose blobs                                                     #
    ###############################################################################
    ksize = (15, 15)
    kernel = getStructuringElement(MORPH_ELLIPSE, ksize)
    channel_A = morphologyEx(channel_A, MORPH_TOPHAT, kernel, iterations=3)
    channel_T = morphologyEx(channel_T, MORPH_TOPHAT, kernel, iterations=3)
    channel_C = morphologyEx(channel_C, MORPH_TOPHAT, kernel, iterations=3)
    channel_G = morphologyEx(channel_G, MORPH_TOPHAT, kernel, iterations=3)
    ########

    ###############################
    # Block of alternative option #
    ###############################
    # channel_A = convertScaleAbs(Laplacian(GaussianBlur(channel_A, (3, 3), 0), CV_32F))
    # channel_T = convertScaleAbs(Laplacian(GaussianBlur(channel_T, (3, 3), 0), CV_32F))
    # channel_C = convertScaleAbs(Laplacian(GaussianBlur(channel_C, (3, 3), 0), CV_32F))
    # channel_G = convertScaleAbs(Laplacian(GaussianBlur(channel_G, (3, 3), 0), CV_32F))
    ###############################

    ###############################################################################
    if temp_flag == "Yes":
        cv2.imwrite(os.path.join(cycle_path, "TOPHAT_A.PNG"), channel_A)
        cv2.imwrite(os.path.join(cycle_path, "TOPHAT_T.PNG"), channel_T)
        cv2.imwrite(os.path.join(cycle_path, "TOPHAT_C.PNG"), channel_C)
        cv2.imwrite(os.path.join(cycle_path, "TOPHAT_G.PNG"), channel_G)

    channel_list = (channel_A, channel_T, channel_C, channel_G)

    mor_kps1 = []
    mor_kps2 = []

    ##########################################################
    # Parameters setup for preliminary blob detection        #
    # Here, some of parameters are very crucial, such as     #
    # 'thresholdStep', 'minRepeatability', 'minArea', which  #
    # could greatly affect the number of detected blobs. And #
    # more importantly, they would need to be modified in    #
    # different experiments.                                 #
    #                                                        #
    # We prepared some cases for the different experiments   #
    # we met during debugging, and we look forward to        #
    # standardize the experiments                            #
    ##########################################################
    blob_params = SimpleBlobDetector_Params()

    blob_params.thresholdStep = 2
    blob_params.minRepeatability = 2
    ########
    # blob_params.thresholdStep = 3  # Alternative option
    # blob_params.minRepeatability = 3  # Alternative option

    blob_params.minDistBetweenBlobs = 2

    ####################################################################################
    # This parameter is used for filtering those extremely large blobs, which likely   #
    # to results from contamination                                                    #
    #                                                                                  #
    # Unfortunately, some genes expressing highly at a dense region tend to form large #
    # blobs thus would to be filtered, lead to optics-identification failure.          #
    # A known case in our practise is the gene 'pro-corazonin-like', this is a highly  #
    # expressed gene at a small region in brain of some insects, and is usually        #
    # detected as a low- or non-expression gene in IRIS                                #
    ####################################################################################
    blob_params.filterByArea = True

    blob_params.minArea = 1
    ########
    # blob_params.minArea = 4  # Alternative option
    blob_params.maxArea = int(float(blob_params_detect["maxArea"]))
    ########
    # blob_params.maxArea = 65  # Alternative option
    # blob_params.maxArea = 145  # Alternative option
    ####################################################################################

    blob_params.filterByCircularity = False
    blob_params.filterByConvexity = True

    blob_params.filterByColor = True
    ##########################################################

    for img in channel_list:
        #################################
        # Setup threshold of gray-scale #
        #################################
        if blob_params_detect["minThreshold"] == "AUTO":
            blob_params.minThreshold = mode(floor(reshape(img, (img.size,)) / 2) * 2)[0][0]
        else:
            blob_params.minThreshold = int(float(blob_params_detect["minThreshold"]))
        #################################
        blob_params.blobColor = 0

        mor_detector = SimpleBlobDetector.create(blob_params)
        mor_kps1.extend(mor_detector.detect(255 - img))

        blob_params.blobColor = 255

        mor_detector = SimpleBlobDetector.create(blob_params)
        mor_kps2.extend(mor_detector.detect(img))

    mor_kps = mor_kps1 + mor_kps2

    #################################################################################
    # To map all the detected blobs into a new mask layer for redundancy filtering, #
    # and detect on this mask layer again to ensure blobs' location across all      #
    # channels in this cycle                                                        #
    #################################################################################
    mask_layer = zeros(channel_A.shape, dtype=uint8)

    for key_point in mor_kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        mask_layer[r:(r + 2), c:(c + 2)] = 255

    #############################################################################################################
    # If a Gaussian blur employed in here, these large blobs could be detected well, but high density blobs not #
    #############################################################################################################
    mask_layer = GaussianBlur(mask_layer, (3, 3), 0)
    #############################################################################################################

    detector = SimpleBlobDetector.create(blob_params)
    kps = detector.detect(mask_layer)
    #################################################################################

    #########################################################################
    # Calculate the threshold for distinction between blobs and potential   #
    # pseudo-blobs                                                          #
    #                                                                       #
    # A crucial feature of real blob is that the gray-scale of pixel        #
    # should increase rapidly in its core region, compared with periphery   #
    #                                                                       #
    # The step of detection could expose a massive amount of blobs but also #
    # include some false-positive. We calculate the difference of mean      #
    # gray-scale between pixel in core region and periphery of each blob,   #
    # which named as 'base score', and calculate a threshold of each        #
    # channel. This threshold could be used to filter those false-positive  #
    # blobs in following step                                               #
    #########################################################################
    diff_list_A = []
    diff_list_T = []
    diff_list_C = []
    diff_list_G = []

    for key_point in kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        diff_A = sum(channel_A[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                 sum(channel_A[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100

        diff_T = sum(channel_T[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                 sum(channel_T[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100

        diff_C = sum(channel_C[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                 sum(channel_C[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100

        diff_G = sum(channel_G[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                 sum(channel_G[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100

        if diff_A > 0:
            diff_list_A.append(int(around(diff_A)))

        if diff_T > 0:
            diff_list_T.append(int(around(diff_T)))

        if diff_C > 0:
            diff_list_C.append(int(around(diff_C)))

        if diff_G > 0:
            diff_list_G.append(int(around(diff_G)))

    diff_bk = 10
    flag_A, flag_T, flag_C, flag_G = True, True, True, True
    if len(diff_list_A) == 0:   flag_A = False
    if len(diff_list_T) == 0:   flag_T = False
    if len(diff_list_C) == 0:   flag_C = False
    if len(diff_list_G) == 0:   flag_G = False

    if flag_A:
        cut_off_A = int(mode(around(divide(array(diff_list_A, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk * 2
    if flag_T:
        cut_off_T = int(mode(around(divide(array(diff_list_T, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk * 2
    if flag_C:
        cut_off_C = int(mode(around(divide(array(diff_list_C, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk * 2
    if flag_G:
        cut_off_G = int(mode(around(divide(array(diff_list_G, dtype=uint8), diff_bk)) * diff_bk)[0][0]) - diff_bk * 2
    #########################################################################

    ##############################################################################################################
    # The coordinates of real blobs will be used to locate the difference of gary-scale among different channels #
    ##############################################################################################################
    for key_point in kps:
        r = int(key_point.pt[1])
        c = int(key_point.pt[0])

        if flag_A:
            if sum(channel_A[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                    sum(channel_A[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100 > cut_off_A:
                greyscale_model_A[r, c] = sum(channel_A[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                                          sum(channel_A[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100
        if flag_T:
            if sum(channel_T[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                    sum(channel_T[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100 > cut_off_T:
                greyscale_model_T[r, c] = sum(channel_T[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                                          sum(channel_T[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100
        if flag_C:
            if sum(channel_C[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                    sum(channel_C[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100 > cut_off_C:
                greyscale_model_C[r, c] = sum(channel_C[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                                          sum(channel_C[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100
        if flag_G:
            if sum(channel_G[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                    sum(channel_G[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100 > cut_off_G:
                greyscale_model_G[r, c] = sum(channel_G[(r - 1):(r + 3), (c - 1):(c + 3)]) / 16 - \
                                          sum(channel_G[(r - 4):(r + 6), (c - 4):(c + 6)]) / 100
    ##############################################################################################################

    image_model_pool = image_model_pooling_Ke(greyscale_model_A,
                                              greyscale_model_T,
                                              greyscale_model_C,
                                              greyscale_model_G)

    base_box_in_one_cycle, base_point = pool2base(image_model_pool)

    # drawing point in each channel
    channel_point(base_point["A"], greyscale_model_A, os.path.join(cycle_path, "A.PNG"))
    channel_point(base_point["T"], greyscale_model_T, os.path.join(cycle_path, "T.PNG"))
    channel_point(base_point["C"], greyscale_model_C, os.path.join(cycle_path, "C.PNG"))
    channel_point(base_point["G"], greyscale_model_G, os.path.join(cycle_path, "G.PNG"))

    return base_box_in_one_cycle


if __name__ == '__main__':
    pass
