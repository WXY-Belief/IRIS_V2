#!/usr/bin/env python3
"""
This module is used to register images which contained in pixel-matrix.

The process of registration can be split into 3 steps:
1) Detect raw key points.
2) Filter out bad matched key point pairs and keep the good ones.
3) Compute transform matrix used in images registration between different cycles.

Here, we used two algorithms for key points detecting: BRISK (S Leutenegger. et al., IEEE, 2011) and
ORB (E Rublee. et al., Citeseer, 2011). The bad matched key points would be marked, and be filtered subsequently to
ensure the accuracy in transform matrix calculation

Transform matrices will be used to transform images by rigid registration. It means there are only translation
and rotation between images but no zooming and retortion.
"""

import cv2
from sys import stderr
from cv2 import (convertScaleAbs,
                 BRISK, ORB, BFMatcher, estimateAffinePartial2D,
                 NORM_HAMMING, RANSAC)
from numpy import (array, zeros, mean, float32, bool_, fft, abs, max)


##########################
# For alternative option #
##########################
# from cv2 import (GaussianBlur, resize, getStructuringElement, morphologyEx,
#                  MORPH_RECT, MORPH_GRADIENT)
# from numpy import around
##########################


def register_cycles(reference_cycle, transform_cycle, detection_method=None):
    """
    For computing the transform matrix between reference image and transform image.

    Input reference image, transform image and one of the algorithms of detector.
    Returning transform matrix.

    :param reference_cycle: Image reference that will be used to register other images.
    :param transform_cycle: Images will be registered.
    :param detection_method: The detection algorithm of feature points.
    :return f_key_points, f_descriptions: A transformation matrix from transformed image to reference.
    """
    def __lpf(f_img):
        """
        Low-pass Filter

        :param f_img: Input image
        :return: Filtered image
        """
        row, col = f_img.shape

        masker_window = zeros((row, col), dtype=bool_)
        masker_window[int(row / 2) - int(row * 0.3):int(row / 2) + int(row * 0.3),
                      int(col / 2) - int(col * 0.3):int(col / 2) + int(col * 0.3)] = 1

        f_img = abs(fft.ifft2(fft.ifftshift(fft.fftshift(fft.fft2(f_img)) * masker_window)))
        f_img = convertScaleAbs(f_img / max(f_img) * 255)

        return f_img

    def __get_key_points_and_descriptors(f_gray_image, method=None):
        """
        For detecting the key points and their descriptions by BRISK or ORB.

        Here, we employed morphology transforming to pre-process image for exposing the key points
        (by a 3x3 Gaussian blur), under a kernel of 15x15. A BRISK or ORB detector used to scan the image for locating
        key points, and computed their descriptions.

        Input a gray scale image and one of the algorithms of detector.
        Returning key points and their descriptions.

        :param f_gray_image: The 8-bit image.
        :param method: The detection algorithm of key points.
        :return: A tuple including a group of key points and their descriptions.
        """
        #################################################################
        # Low-pass filter in frequency domain of Fourier transformation #
        #################################################################
        f_gray_image = __lpf(f_gray_image)
        ########

        ###########################################################################
        # Block of alternative option                                             #
        #                                                                         #
        # In order to reduce the errors better in registration, we need to reduce #
        # some redundant features in each image. Here, a method of morphological  #
        # transformation, Morphological gradient, the difference between          #
        # dilation and erosion of an image, is used to expose key points under a  #
        # 15x15 rectangle kernel, after a Gaussian blur (3x3 kernel).             #
        ###########################################################################
        # f_gray_image = GaussianBlur(f_gray_image, (3, 3), 0)
        # ksize = (15, 15)
        # kernel = getStructuringElement(MORPH_RECT, ksize)
        # f_gray_image = morphologyEx(f_gray_image, MORPH_GRADIENT, kernel, iterations=2)
        ###########################################################################

        #################################################################

        det = ''
        ext = ''

        ##############################################################################################
        # We prepare two methods of feature points detection for selectable, one is 'BRISK', and the #
        # other is 'ORB'. In general, the algorithm 'ORB' is used as the open-source alternative of  #
        # 'SIFT' and 'SURF', which are almost the industry standard. In our practice, 'ORB' always   #
        # detect very fewer but more robust key points than 'BRISK' and often lead to registration   #
        # failed. So, we choose the latter as our method of point detecting, in default.             #
        ##############################################################################################
        method = 'ORB' if method is None else method

        if method == 'ORB':
            det = ORB.create()
            ext = ORB.create()

        elif method == 'BRISK':
            det = BRISK.create()
            ext = BRISK.create()
        elif method == 'SIFT':
            det = cv2.SIFT_create()
            ext = cv2.SIFT_create()
        elif method == 'BLOB':
            det = cv2.SimpleBlobDetector_create()
            # ext = cv2.SimpleBlobDetector_create()
        elif method == 'GFTT':
            det = cv2.GFTTDetector_create(maxCorners=5000, qualityLevel=0.01, minDistance=3.0,
                                          blockSize=3, useHarrisDetector=True, k=0.04)
            ext = ORB.create()
        else:
            print('Only ORB or BRISK could be suggested', file=stderr)
        ##############################################################################################

        # f_key_points = det.detect(f_gray_image)
        # _, f_descriptions = ext.compute(f_gray_image, f_key_points)

        f_key_points, f_descriptions = det.detectAndCompute(f_gray_image, None)
        return f_key_points, f_descriptions

    def __get_good_matched_pairs(f_description1, f_description2):
        """
        For finding good matched key point pairs.

        The matched pairs of key points would be filtered to generate a group of good matched pairs.
        These good matched pairs of key points would be used to compute the transform matrix.

        Input two groups of description.
        Returning the good matched pairs of key points.

        :param f_description1: The description of feature points group 1.
        :param f_description2: The description of feature points group 2.
        :return f_good_matched_pairs: The good matched pairs between those two groups of feature points.
        """
        # matcher = BFMatcher.create(normType=NORM_HAMMING, crossCheck=True)
        #
        # matched_pairs = matcher.knnMatch(f_description1, f_description2, 1)
        #
        # f_good_matched_pairs = [best_match_pair[0] for best_match_pair in matched_pairs if len(best_match_pair) > 0]
        # f_good_matched_pairs = sorted(f_good_matched_pairs, key=lambda x: x.distance)
        #
        # return f_good_matched_pairs
        try :
            matcher = BFMatcher.create(normType=NORM_HAMMING, crossCheck=True)
        #print(len(f_description1))
        #print(len(f_description2))
            matched_pairs = matcher.knnMatch(f_description1, f_description2, 1) #计算图片匹配数)
            f_good_matched_pairs = [best_match_pair[0] for best_match_pair in matched_pairs if len(best_match_pair) > 0]
            f_good_matched_pairs = sorted(f_good_matched_pairs, key=lambda x: x.distance)
        except :
            index_params=dict(algorithm=1,trees=5)
            search_params=dict(checks=50)
            flann=cv2.FlannBasedMatcher(index_params,search_params)
            matched_pairs=flann.knnMatch(f_description1.astype('float32'),f_description2.astype('float32'),k=2)

        #f_good_matched_pairs = [best_match_pair[0] for best_match_pair in matched_pairs if len(best_match_pair) > 0] #提取每个位点最佳匹配位置
        #f_good_matched_pairs = sorted(f_good_matched_pairs, key=lambda x: x.distance)
            f_good_matched_pairs = []
            for m,n in matched_pairs:
                if m.distance < 0.7*n.distance:
                    f_good_matched_pairs.append(m)

        return f_good_matched_pairs

    ################

    transform_matrix = array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float32)

    #######################################
    # Lightness Rectification (IMPORTANT) #
    #######################################
    transform_cycle = convertScaleAbs(transform_cycle * (mean(reference_cycle) / mean(transform_cycle)))
    #######################################

    kp1, des1 = __get_key_points_and_descriptors(reference_cycle, detection_method)
    kp2, des2 = __get_key_points_and_descriptors(transform_cycle, detection_method)

    good_matches = __get_good_matched_pairs(des1, des2)

    #################################################################################
    # Filter the outline of paired key points iteratively until there's no outlines #
    #################################################################################
    n = 1
    while n > 0:
        pts_a = float32([kp1[_.queryIdx].pt for _ in good_matches]).reshape(-1, 1, 2)
        pts_b = float32([kp2[_.trainIdx].pt for _ in good_matches]).reshape(-1, 1, 2)

        _, mask = estimateAffinePartial2D(pts_b, pts_a)

        good_matches = [good_matches[_] for _ in range(0, mask.size) if mask[_][0] == 1]

        n = sum([mask[_][0] for _ in range(0, mask.size)]) - mask.size
    ###########################################################################

    if len(good_matches) >= 4:
        pts_a_filtered = float32([kp1[_.queryIdx].pt for _ in good_matches]).reshape(-1, 1, 2)
        pts_b_filtered = float32([kp2[_.trainIdx].pt for _ in good_matches]).reshape(-1, 1, 2)

        transform_matrix, _ = estimateAffinePartial2D(pts_b_filtered, pts_a_filtered, RANSAC)

        if transform_matrix is None:
            print('MATRIX GENERATION FAILED.', file=stderr)

    else:
        print('NO ENOUGH MATCHED FEATURES, REGISTRATION FAILED.', file=stderr)

    return transform_matrix


if __name__ == '__main__':
    pass
