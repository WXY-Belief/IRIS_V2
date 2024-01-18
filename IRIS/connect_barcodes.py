#!/usr/bin/env python3
"""
This module is used to connect bases from different cycles in the same location to form barcode sequence.

In this model, the connection is made as a class, the 'BarcodeCube', of which, three method will be employed. The
method 'collect_called_bases' is used to record the called bases in each cycle, and 'filter_blobs_list' is used to
filter the bad or indistinguishable blobs by mapping them into a mask layer. At last, the method 'calling_adjust' is
used to connect the bases as barcodes, by anchoring the coordinates of blobs in reference layer, and search their 6x6
region in each cycle.
"""

from sys import stderr
from cv2 import (SimpleBlobDetector_Params, SimpleBlobDetector, GaussianBlur)
from numpy import (sqrt, zeros, uint8)
import math

class BarcodeCube:
    def __init__(self, search_region=2):
        """
        This method will initialize three members, '__all_blobs_list' store all blobs' id, 'bases_cube' is
        a list stores the dictionary of bases in each cycle, and 'adjusted_bases_cube' is a list stores
        the dictionary of bases in each cycle, with error rate adjusted.
        """
        self.__all_blobs_list = []

        self.bases_cube = []
        self.adjusted_bases_cube = []

        ########################################################################################################
        # Setup search region                                                                                  #
        # The more large you setup, the more well in TPR and correlation with RNA-Seq, but less number of blob #
        ########################################################################################################
        # self.__search_region = search_region  # 6x6
        if type(search_region) == int:
            self.left_search_region = search_region
            self.right_search_region = search_region+2
        else:
            self.left_search_region = math.ceil(search_region)
            self.right_search_region = math.ceil(search_region)+1
        ########
        # self.__search_region = 0  # Alternative option, 2x2
        # self.__search_region = 1  # Alternative option, 4x4
        # self.__search_region = 3  # Alternative option, 8x8
        # self.__search_region = 4  # Alternative option, 10x10
        # self.__search_region = n  # Alternative option, ((n + 1) * 2)x((n + 1) * 2)
        ########################################################################################################

    def collect_called_bases(self, called_base_in_one_cycle):
        """
        This method is used to record the called bases in each cycle.

        A list which store all blobs' id and a list which store the dictionary of bases in each cycle.

        :param called_base_in_one_cycle: The dictionary of bases in a cycle.
        :return: NONE
        """
        self.__all_blobs_list.extend([_ for _ in called_base_in_one_cycle.keys()])
        self.bases_cube.append(called_base_in_one_cycle)

    #################################
    # Redundancy-filtering strategy #
    #################################
    def filter_blobs_list(self, f_background):
        """
        This method is used to filter the recorded bases in the called base list.

        A new list will be generated, which store the filtered id of bases

        :param f_background: The background image for ensuring the shape of mask layer.
        :return: NONE
        """
        blobs_mask = zeros(f_background.shape, dtype=uint8)

        new_coor = set()

        for coor in self.__all_blobs_list:
            r = int(coor[1:6].lstrip('0'))
            c = int(coor[7:].lstrip('0'))

            blobs_mask[r, c] = 255

        blobs_mask = GaussianBlur(blobs_mask, (3, 3), 0)

        blob_params = SimpleBlobDetector_Params()

        blob_params.thresholdStep = 1
        blob_params.minRepeatability = 1
        blob_params.minDistBetweenBlobs = 2

        blob_params.filterByArea = True
        blob_params.minArea = 1
        blob_params.maxArea = 65

        blob_params.filterByCircularity = False
        blob_params.filterByConvexity = False

        blob_params.filterByColor = True

        blob_params.blobColor = 0

        detector = SimpleBlobDetector.create(blob_params)
        kps = detector.detect(255 - blobs_mask)

        for key_point in kps:
            r = int(key_point.pt[1])
            c = int(key_point.pt[0])

            new_coor.add(str('r' + ('%05d' % r) + 'c' + ('%05d' % c)))

        self.__all_blobs_list = new_coor

    ########

    ###############################
    # Block of alternative option #
    ###############################
    def filter_blobs_list2(self):
        """
        This method is used to filter the recorded bases in the called base list.

        A new list will be generated, which store the filtered id of bases
        """
        new_coor = self.__all_blobs_list

        for coor in self.__all_blobs_list:
            r = int(coor[1:6].lstrip('0'))
            c = int(coor[7:].lstrip('0'))

            for row in range(r, r + 2):
                for col in range(c, c + 2):
                    if row == r and col == c:
                        continue

                    if 'r%05dc%05d' % (row, col) in self.__all_blobs_list:
                        new_coor.remove('r%05dc%05d' % (row, col))

        self.__all_blobs_list = set(new_coor)

    ###############################

    #################################

    def calling_adjust(self):
        """
        This method is used to connect bases into barcodes, by anchoring the coordinates of blobs in reference layer,
        and search their NxN region in each cycle.

        :return: NONE
        """

        def __check_greyscale(all_blobs_list, bases_cube, adjusted_bases_cube, cycle_serial):
            """"""
            adjusted_bases_cube[cycle_serial] = {}
            temp_1 = []
            for ref_coordinate in all_blobs_list:
                r = int(ref_coordinate[1:6].lstrip('0'))
                c = int(ref_coordinate[7:].lstrip('0'))

                max_qul_base = 'N'
                min_err_rate = float(1)

                ##################################################################################################
                # It will search a NxN region to connect bases from each cycle in ref-coordinates                #
                #                                                                                                #
                # Process of registration almost align all location of cycles the same, but at pixel level, this #
                # registration is not accurate enough. Here, we choose a simple approach to solve this problem,  #
                # we get locations of blobs from a reference image layer, then to search a NxN (6x6 by default)  #
                # region in those cycles that need to be connected. This approach should not only solve this     #
                # problem but also bring few false positive in output                                            #
                ##################################################################################################
                haha = "N"
                for row in range(r - self.left_search_region, r + self.right_search_region):
                    for col in range(c - self.left_search_region, c + self.right_search_region):
                        coor = 'r%05dc%05d' % (row, col)

                        if coor in bases_cube[cycle_serial]:
                            error_rate = bases_cube[cycle_serial][coor][1]

                            ##############################################################
                            # Adjust of error rate of each coordinate by the Pythagorean #
                            # theorem. This function can be off if no need               #
                            #
                            # If the search region is so large you set, we suggest you   #
                            # to adjust the error rate based on Pythagorean theorem      #
                            ##############################################################
                            D = sqrt((row - r) ** 2 + (col - c) ** 2)
                            adj_err_rate = sqrt(((error_rate * D) ** 2) + (error_rate ** 2))
                            ########
                            # adj_err_rate = error_rate  # Alternative option
                            ##############################################################

                            if adj_err_rate > 1:
                                adj_err_rate = float(1)

                            if adj_err_rate < min_err_rate:
                                max_qul_base = bases_cube[cycle_serial][coor][0]
                                min_err_rate = adj_err_rate
                                haha = coor
                ##################################################################################################
                adjusted_bases_cube[cycle_serial].update({ref_coordinate: [max_qul_base, min_err_rate, haha]})

        if len(self.bases_cube) > 0:
            for cycle_id in range(0, len(self.bases_cube)):
                self.adjusted_bases_cube.append({})

                __check_greyscale(set(self.__all_blobs_list), self.bases_cube, self.adjusted_bases_cube, cycle_id)

            if len(self.bases_cube) == 1:
                print('There is only one cycle in this run', file=stderr)


if __name__ == '__main__':
    pass
