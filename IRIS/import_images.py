#!/usr/bin/env python3
"""
This model is used to import the images and store them into a 3D matrix.
Our software generate a 3D matrix to store all the images. Each channel is made of a image matrix, and insert into this
tensor in the order of cycle
"""
import os.path
import numpy as np
from IRIS.common_class import read_image
from sys import stderr
from cv2 import (imwrite, add, addWeighted, warpAffine)
from numpy import (array, uint8)
from .register_images import register_cycles


def decode_data_Ke(config, data_path, output_path, channel):
    """
    :param f_cycles: The image directories in sequence of cycles, of which the different channels are stored.
    :return: A tuple including a 3D matrix and a background image matrix.

    """
    f_cycles = config["cycle"]
    register_mode = config["register_mode"]
    temp_flag = int(config["temp_flag"])
    debug_save_path = output_path

    if len(f_cycles) < 1:
        print('ERROR CYCLES', file=stderr)
        exit(1)

    f_cycle_stack = []
    f_std_img = array([], dtype=uint8)
    reg_ref = array([], dtype=uint8)

    for cycle_id in range(0, len(f_cycles)):
        adj_img_mats = []

        ####################################
        # Read five channels into a matrix #
        ####################################
        channel_A = read_image(os.path.join(data_path, str(f_cycles[cycle_id]), channel["A"]))
        channel_T = read_image(os.path.join(data_path, str(f_cycles[cycle_id]), channel["T"]))
        channel_C = read_image(os.path.join(data_path, str(f_cycles[cycle_id]), channel["C"]))
        channel_G = read_image(os.path.join(data_path, str(f_cycles[cycle_id]), channel["G"]))
        channel_0 = read_image(os.path.join(data_path, str(f_cycles[cycle_id]), channel[0]))
        ####################################

        #########################################################################################
        # Merge different channels from a same cycle into one matrix for following registration #
        #                                                                                       #
        # BE CARE: The parameters 'alpha' and 'beta' maybe will affect whether the registering  #
        # success. Sometimes, a registration would succeed with only using DAPI from different  #
        # cycle instead of merged images                                                        #
        #########################################################################################
        merged_img = channel_0
        ########

        ###############################
        # Block of alternative option #
        ###############################
        # alpha = 0.5
        # beta = 0.6
        # merged_img = addWeighted(add(add(add(channel_A, channel_T), channel_C), channel_G), alpha, channel_0, beta, 0)
        ###############################

        if cycle_id == 0:
            reg_ref = merged_img

            ###################################
            # Output background independently #
            ###################################
            foreground = add(add(add(channel_A, channel_T), channel_C), channel_G)
            background = channel_0

            f_std_img = addWeighted(foreground, 0.5, background, 0.6, 0)
            ########
            # f_std_img = foreground
            # f_std_img = addWeighted(foreground, 0.5, background, 0.5, 0)  # Alternative option
            # f_std_img = addWeighted(foreground, 0.4, background, 0.8, 0)  # Alternative option
            ###################################

        trans_mat = register_cycles(reg_ref, merged_img, register_mode)

        #############################

        channel_A = warpAffine(channel_A, trans_mat, (reg_ref.shape[1], reg_ref.shape[0]))
        channel_T = warpAffine(channel_T, trans_mat, (reg_ref.shape[1], reg_ref.shape[0]))
        channel_C = warpAffine(channel_C, trans_mat, (reg_ref.shape[1], reg_ref.shape[0]))
        channel_G = warpAffine(channel_G, trans_mat, (reg_ref.shape[1], reg_ref.shape[0]))
        channel_0 = warpAffine(channel_0, trans_mat, (reg_ref.shape[1], reg_ref.shape[0]))

        adj_img_mats.append(channel_A)
        adj_img_mats.append(channel_T)
        adj_img_mats.append(channel_C)
        adj_img_mats.append(channel_G)
        adj_img_mats.append(channel_0)
        #########################################################################################

        ###################################################################################################
        # This stacked 3D-tensor is a common data structure for following analysis and data compatibility #
        ###################################################################################################
        f_cycle_stack.append(adj_img_mats)
        ###################################################################################################

        #############################
        # For registration checking #
        #############################
        if temp_flag:
            debug_img = np.zeros((channel_0.shape[0], channel_0.shape[1], 3))
            debug_img[:, :, 0] = f_cycle_stack[0][-1]
            debug_img[:, :, 1] = channel_0

            imwrite(os.path.join(debug_save_path,
                                 'debug_cycle_reg' + str(f_cycles[0]) + "_this_" + str(f_cycles[cycle_id]) + '.PNG'),
                    debug_img)
    imwrite(os.path.join(output_path, 'background.tif'), f_std_img)
    return f_cycle_stack, f_std_img


if __name__ == '__main__':
    pass
