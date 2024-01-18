import os
import yaml
import time
import shutil
import argparse
import numpy as np
from sys import stderr
from numpy import (array, uint8)
from cv2 import (imread, add, addWeighted, IMREAD_GRAYSCALE)
from multiprocessing import Pool
from IRIS import (import_images, detect_signals, connect_barcodes, deal_with_result)
from IRIS.common_class import CutImages, TransformCoordinate
from IRIS.count_multiplex_base_ID import count_multiplex_base_id, count_multiplex_point


# Reading yaml file
def load_config_file(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data


# Importing images
def import_img(f_cycles):
    if len(f_cycles) < 1:
        print('ERROR CYCLES', file=stderr)
        exit(1)

    f_cycle_stack = []
    f_std_img = array([], dtype=uint8)

    for cycle_id in range(0, len(f_cycles)):
        adj_img_mats = []
        channel_A = imread('/'.join((str(f_cycles[cycle_id]), 'Cy5.tif')), IMREAD_GRAYSCALE)
        channel_T = imread('/'.join((str(f_cycles[cycle_id]), 'Y7.tif')), IMREAD_GRAYSCALE)
        channel_C = imread('/'.join((str(f_cycles[cycle_id]), 'TXR.tif')), IMREAD_GRAYSCALE)
        channel_G = imread('/'.join((str(f_cycles[cycle_id]), 'Cy3.tif')), IMREAD_GRAYSCALE)
        channel_0 = imread('/'.join((str(f_cycles[cycle_id]), 'DAPI.tif')), IMREAD_GRAYSCALE)

        if cycle_id == 0:
            foreground = add(add(add(channel_A, channel_T), channel_C), channel_G)
            background = channel_0
            f_std_img = addWeighted(foreground, 0.5, background, 0.6, 0)

        adj_img_mats.append(channel_A)
        adj_img_mats.append(channel_T)
        adj_img_mats.append(channel_C)
        adj_img_mats.append(channel_G)
        adj_img_mats.append(channel_0)

        f_cycle_stack.append(adj_img_mats)

    return f_cycle_stack, f_std_img


def main_detect(info: list):
    os.chdir(info[0])  # setting current work dir
    config = load_config_file(info[1])  # load conguration file

    barcode_cube_obj = connect_barcodes.BarcodeCube(config["search_region"])
    cycle_stack, std_img = import_img(config["cycle"])  # import and align images

    if np.sum(std_img) == 0:
        print(info[0], "is completely black")
        return
    for i in range(len(cycle_stack)):
        cycle_path = os.path.join(info[0], str(config["cycle"][i]))

        # detect blob in any channel
        called_base_box_in_one_cycle = detect_signals.detect_blobs_Ke(cycle_stack[i], cycle_path,
                                                                      int(config["temp_flag"]), config["blob_params"])
        barcode_cube_obj.collect_called_bases(called_base_box_in_one_cycle)

    barcode_cube_obj.filter_blobs_list(std_img)  # filtering detected blob
    barcode_cube_obj.calling_adjust()  # connecting bases on all cycle to form gene barcode

    deal_with_result.write_reads_into_file(std_img, barcode_cube_obj.adjusted_bases_cube, len(cycle_stack))
    count_multiplex_base_id(info[0])


def main_call(config, data_path, output_path, config_path):
    print(output_path)
    # aligning tissue slice in all cycle by reference
    time_1 = time.time()

    cycle_stack, std_img = import_images.decode_data_Ke(config, data_path, output_path)
    print("The time aligned images is", int(time.time() - time_1))

    # cuting img
    small_img_para = config["small_img_para"]
    small_images = CutImages(data_path, small_img_para["cut_size"], output_path, small_img_para["overlap"],
                             config["cycle"], config["iris_path"])
    small_images.main_cut(cycle_stack)
    print("the number of small img is", len(small_images.infos))

    # detected RNA
    time_2 = time.time()
    pool = Pool(processes=core_num)
    pool.map(main_detect, [[_, config_path] for _ in small_images.infos])
    pool.close()
    pool.join()
    print("The time detected RNA is", int(time.time() - time_2))

    # stitching img and transforming coordinate
    barcode_path = config["barcode_path"]
    small_stitch = TransformCoordinate(os.path.join(output_path, "small_vision"), barcode_path, output_path,
                                       small_img_para["cut_size"], small_img_para["overlap"])
    small_stitch.transform_coordinate()

    # copy configuration into putput dir
    shutil.copyfile(config_path, os.path.join(output_path, "Configuration.yaml"))


if __name__ == '__main__':
    total_start_time = time.time()

    # load configuration file
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--c', default="./Configuration.yaml")
    args = parser.parse_args()

    config = load_config_file(args.c)
    core_num = config["core_num"]
    cut_mode = config["mode"]
    output_path = os.path.join(config["output_path"],
                               config["register_mode"] + "_" + str(config["search_region"]) + "_output_result")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    # whether to cut images when images is too large.  com:NO large:YES
    if cut_mode == "large":
        large_img_para = config["large_img_para"]
        Big_images = CutImages(config["data_path"], large_img_para["cut_size"], output_path, large_img_para["overlap"],
                               config["cycle"], config["iris_path"], cut_mode)
        Big_images.main_cut()
        for item in Big_images.infos:
            print(f"--------------------{item}-------------------")
            single_output_path = os.path.join(output_path, item.split("/")[-1])
            os.makedirs(single_output_path, exist_ok=True)
            main_call(config, item, single_output_path, args.c)
        Big_stitch = TransformCoordinate(output_path, config["barcode_path"], output_path, large_img_para["cut_size"],
                                         large_img_para["overlap"])
        Big_stitch.transform_coordinate()

    else:
        main_call(config, config["data_path"], output_path, args.c)

    # counting the number of multiplex
    count_multiplex_point(output_path)
    print("The total time is", int(time.time() - total_start_time))
