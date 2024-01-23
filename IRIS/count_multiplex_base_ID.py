import pandas as pd
import cv2
import os
import csv


def draw_point(multiplex_base_ID_point, rest_valid_point, save_path):
    img = cv2.imread(os.path.join(save_path, "background.tif"))
    for idx, item in multiplex_base_ID_point.iterrows():
        cv2.circle(img, (int(item["col"]), int(item["row"])), radius=3, color=(0, 255, 0),
                   thickness=-1)
    for idx, item in rest_valid_point.iterrows():
        cv2.circle(img, (int(item["col"]), int(item["row"])), radius=3, color=(0, 0, 255),
                   thickness=-1)

    cv2.imwrite(os.path.join(save_path, "multiplex_base_ID.PNG"), img)


def count_multiplex_base_id(save_path):
    data = pd.read_csv(os.path.join(save_path, "basecalling_data.txt"), sep="\t", header=0, quoting=csv.QUOTE_NONE)

    final_mul_id = []
    for item in ['1', '2', '3', '4']:
        mul_id = []
        haha = data[["ID", item]]
        haha = haha[haha[item] != "N"]
        only_id = haha[item].drop_duplicates().tolist()

        for item_2 in only_id:
            repete_id = haha[haha[item] == item_2]
            if repete_id.shape[0] > 1:
                mul_id.extend(repete_id.iloc[1:]["ID"].tolist())
        final_mul_id.extend(mul_id)

    final_mul_ID = list(set(final_mul_id))
    data["mul_flag"] = 0
    data.loc[data['ID'].isin(final_mul_ID), 'mul_flag'] = 1
    data.to_csv(os.path.join(save_path, "mul_basecalling_data.txt"), sep="\t", header=True, index=False)


def count_multiplex_point(path, RNA_filter_flag):
    if RNA_filter_flag == "Yes":
        data = pd.read_csv(os.path.join(path, "valid_rna_coordinate.csv"), sep=",", header=0)

        mul_id = data[data["mul_flag"] == 1]
        mul_id.to_csv(os.path.join(path, "multiplex_point.csv"), sep=",", header=True, index=False)
        print("the total number of valid multiplex_base_id:", mul_id.shape[0])

        rest_point = data.drop(mul_id.index, axis=0)
        print("the total number of normal point:", rest_point.shape[0])
        draw_point(data, rest_point, path)
    else:
        print("No filtering was performed")


if __name__ == "__main__":
    pass
