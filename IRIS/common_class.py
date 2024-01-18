import numpy as np
from PIL import Image
import os
import cv2
import math
import shutil
import csv
import pandas as pd
from cv2 import add, addWeighted


# cut images
class CutImages(object):
    def __init__(self, images_path, cut_size, save_path, overlap, cycle_names, iris_path, mode="com"):
        self.images_path = images_path
        self.cut_size = cut_size
        self.save_path = save_path
        self.mode = mode
        self.overlap = overlap
        self.cycle_names = cycle_names
        self.iris_path = iris_path
        self.row_num = 0
        self.col_num = 0
        self.infos = []

    def calculate_row_num_and_col_num(self):
        images = Image.open(os.path.join(self.images_path, str(self.cycle_names[0]), "DAPI.tif"))
        self.row_num = math.ceil(images.height / (self.cut_size - self.overlap))
        self.col_num = math.ceil(images.width / (self.cut_size - self.overlap))

    def mkdir_dir(self):
        if self.mode == "com":
            if os.path.exists(os.path.join(self.save_path, "small_vision")):
                shutil.rmtree(os.path.join(self.save_path, "small_vision"))

        for i in range(self.row_num):
            for j in range(self.col_num):
                if self.mode == "large":
                    base_dir = os.path.join(self.save_path, str(i) + "_" + str(j))
                    os.makedirs(base_dir, exist_ok=True)
                elif self.mode == "com":
                    base_dir = os.path.join(self.save_path, "small_vision", str(i) + "_" + str(j))
                    os.makedirs(base_dir, exist_ok=True)

                    if os.path.exists(os.path.join(base_dir, "IRIS")):
                        shutil.rmtree(os.path.join(base_dir, "IRIS"))
                    shutil.copytree(self.iris_path, os.path.join(base_dir, "IRIS"))
                else:
                    print("cut mode is wrong")
                    return

                self.infos.append(base_dir)

                for k in self.cycle_names:
                    os.makedirs(os.path.join(base_dir, str(k)), exist_ok=True)

    def cut_image(self, image, img_name, cycle_name):
        for row in range(self.row_num):
            for col in range(self.col_num):
                left, top = col * (self.cut_size - self.overlap) - self.overlap, row * (
                        self.cut_size - self.overlap) - self.overlap
                right, bottom = left + self.cut_size, top + self.cut_size
                if left < 0:
                    left = 0
                if top < 0:
                    top = 0
                cropped_image = image[top:bottom, left:right]
                if self.mode == "large":
                    cv2.imwrite(os.path.join(self.save_path, str(row) + "_" + str(col), str(cycle_name),
                                             img_name), cropped_image)

                elif self.mode == "com":
                    cv2.imwrite(os.path.join(self.save_path, "small_vision", str(row) + "_" + str(col), str(cycle_name),
                                             img_name), cropped_image)
                else:
                    print("cut mode is wrong")
                    return

    def main_cut(self, cycle_stack=None):
        # calculating the number of row and col
        self.calculate_row_num_and_col_num()

        # creating folder
        self.mkdir_dir()

        if self.mode == "large":
            for i in range(0, len(self.cycle_names)):
                channel_A = np.array(Image.open(os.path.join(self.images_path, str(self.cycle_names[i]), "Cy5.tif")))
                channel_T = np.array(Image.open(os.path.join(self.images_path, str(self.cycle_names[i]), "Y7.tif")))
                channel_C = np.array(Image.open(os.path.join(self.images_path, str(self.cycle_names[i]), "TXR.tif")))
                channel_G = np.array(Image.open(os.path.join(self.images_path, str(self.cycle_names[i]), "Cy3.tif")))
                channel_0 = np.array(Image.open(os.path.join(self.images_path, str(self.cycle_names[i]), "DAPI.tif")))

                self.cut_image(channel_A, "Cy5.tif", self.cycle_names[i])
                self.cut_image(channel_T, "Y7.tif", self.cycle_names[i])
                self.cut_image(channel_C, "TXR.tif", self.cycle_names[i])
                self.cut_image(channel_G, "Cy3.tif", self.cycle_names[i])
                self.cut_image(channel_0, "DAPI.tif", self.cycle_names[i])

                if self.cycle_names[i] == 1:
                    background_img = add(add(add(channel_A, channel_T), channel_C), channel_G)
                    f_std_img = addWeighted(background_img, 0.5, channel_0, 0.6, 0)

                    cv2.imwrite(os.path.join(self.save_path, "background.tif"), f_std_img)
        else:
            for i in range(0, len(cycle_stack)):
                channel_A = cycle_stack[i][0]
                channel_T = cycle_stack[i][1]
                channel_C = cycle_stack[i][2]
                channel_G = cycle_stack[i][3]
                channel_0 = cycle_stack[i][4]

                self.cut_image(channel_A, "Cy5.tif", self.cycle_names[i])
                self.cut_image(channel_T, "Y7.tif", self.cycle_names[i])
                self.cut_image(channel_C, "TXR.tif", self.cycle_names[i])
                self.cut_image(channel_G, "Cy3.tif", self.cycle_names[i])
                self.cut_image(channel_0, "DAPI.tif", self.cycle_names[i])


# draw point
class DrawPoint(object):
    def __init__(self, points, bg_img_path, is_no_bg: bool, save_path, point_size=3, point_color=(0, 255, 0)):
        self.points = points
        self.bg_img_path = bg_img_path
        self.is_no_bg = is_no_bg
        self.save_path = save_path
        self.point_size = point_size
        self.point_color = point_color

    def draw(self):
        if self.is_no_bg:
            img = cv2.imread(self.bg_img_path)
        else:
            img = np.zeros(cv2.imread(self.bg_img_path).shape)
        for idx, item in self.points.iterrows():
            cv2.circle(img, (int(item["col"]), int(item["row"])), radius=self.point_size, color=self.point_color,
                       thickness=-1)
        cv2.imwrite(os.path.join(self.save_path), img)


# transform coordinate
class TransformCoordinate(object):
    def __init__(self, data_path, barcode_path, save_path, cut_size, overlap):
        self.data_path = data_path
        self.barcode_path = barcode_path
        self.save_path = save_path
        self.cut_size = cut_size
        self.overlap = overlap

    def transform_coordinate(self):
        img_path_and_cycle_name = os.listdir(self.data_path)
        barcode = pd.read_csv(self.barcode_path, sep="\t", header=0, quoting=csv.QUOTE_NONE)

        ori_rna = pd.DataFrame()
        valid_rna = pd.DataFrame()
        for item in img_path_and_cycle_name:
            rna_path = os.path.join(self.data_path, item, "mul_basecalling_data.txt")
            if not os.path.exists(rna_path):
                continue
            rna_coor = pd.read_csv(rna_path, sep="\t", header=0, quoting=csv.QUOTE_NONE)

            row, col = float(item.split("_")[0]), float(item.split("_")[1])
            row_start, col_start = row * (self.cut_size - self.overlap) - self.overlap, col * (
                    self.cut_size - self.overlap) - self.overlap

            if row_start < 0: row_start = 0
            if col_start < 0: col_start = 0

            rna_coor["row"] = rna_coor["row"] + row_start
            rna_coor["col"] = rna_coor["col"] + col_start

            ori_rna = pd.concat([ori_rna, rna_coor])  # all rna
            rna_coor = pd.merge(rna_coor, barcode, on="barcode")
            valid_rna = pd.concat([valid_rna, rna_coor])  # valid rna

        ori_rna.to_csv(os.path.join(self.save_path, "basecalling_data.txt"), sep="\t", header=True, index=False)
        print("The number of all RNA detectedis", ori_rna.shape[0])
        valid_rna.to_csv(os.path.join(self.save_path, "valid_rna_coordinate.csv"), sep=",", header=True, index=False)
        print("The number of valid RNA is", valid_rna.shape[0])

        v_point = DrawPoint(valid_rna, os.path.join(self.save_path, "background.tif"), True,
                            os.path.join(self.save_path, "valid_rna_coordinate.PNG"))
        v_point.draw()
