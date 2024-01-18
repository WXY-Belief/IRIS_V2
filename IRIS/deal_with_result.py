#!/usr/bin/env python3
"""
This model is used to transform error rate into Phred+ 33 score, then output the background and the formatted
result of base calling.
"""

from cv2 import imwrite
from numpy import log10


def write_reads_into_file(f_background, f_barcode_cube, f_barcode_length):
    """
    This function is used to transform error rate into Phred+ 33 score, then output the background and the
    formatted result of base calling.

    :param f_background: The image matrix of background.
    :param f_barcode_cube: The connected barcode, with error rate of each base.
    :param f_barcode_length: The length of barcode.
    :return: NONE
    """
    imwrite('background.tif', f_background)

    with open('basecalling_data.txt', 'wt') as ou:
        print("ID" + '\t' + "barcode" + '\t' + "quanlity" + '\t' + "row" + '\t' + "col" + '\t' + str(1) + '\t' + str(
            2) + '\t' + str(3) + '\t' + str(4), file=ou)
        for j in f_barcode_cube[0]:
            coo = [j[1:6], j[7:]]
            seq = []
            qul = []
            cycle_id = []

            for k in range(0, f_barcode_length):
                if f_barcode_cube[k][j][1] is not None:
                    ###############################################################
                    # Transforming the error rate into the Phred+ 33 score system #
                    # It is also transform to the Phred+ 64 score system if need  #
                    ###############################################################
                    quality = int(-10 * log10(f_barcode_cube[k][j][1] + 0.0001)) + 33
                    ########
                    # quality = int(-10 * log10(f_barcode_cube[k][j][1] + 0.001)) + 64  # Alternative option

                    seq.append(f_barcode_cube[k][j][0])
                    qul.append(chr(quality))
                    cycle_id.append(f_barcode_cube[k][j][2])

            print(j + '\t' + ''.join(seq) + '\t' + ''.join(qul) + '\t' + '\t'.join(coo) + "\t" + "\t".join(cycle_id),
                  file=ou)
