import os
from skimage.io import imsave
import numpy as np
from image_synthesis import Synthesize
import face3d
from face3d.morphable_model import MorphabelModel
from .pos_map_code import process_uv, run_posmap_300W_LP

def generate_300WLP_dataset(root_300wlp_folder="/lhome/yongbk/300W-LP/300W_LP", output_root_folder="/lhome/yongbk/results/300W-LP", file_list_fp="/lhome/yongbk/results/300W-LP/file_list.txt"):
    uv_coords = face3d.morphable_model.load.load_uv_coords("Data/BFM/BFM_UV.mat")
    uv_coords = process_uv(uv_coords)
    bfm = MorphabelModel("Data/BFM/BFM.mat")

    synthesizer = Synthesize()
    directories_to_skip = ["Code", "landmarks"]

    file_list = open(file_list_fp, "w")
    for dataset in list(filter(lambda directory: directory not in directories_to_skip, os.listdir(root_300wlp_folder))):
        if not os.path.exists(f"{output_root_folder}/{dataset}"):
            os.makedirs(f"{output_root_folder}/{dataset}")
        for file in list(filter(lambda file: file.endswith(".jpg"), os.listdir(f"{root_300wlp_folder}/{dataset}"))):
            img_fp = f"{root_300wlp_folder}/{dataset}/{file}"
            mat_fp = os.path.splitext(img_fp)[0]+".mat"
            output_folder = f"{output_root_folder}/{dataset}"
            run_posmap_300W_LP(bfm, uv_coords, img_fp, mat_fp, output_folder)

            save_img_fp = f"{output_root_folder}/{dataset}/{file}"
            save_rotated_img_fp = f"{output_root_folder}/{dataset}/{os.path.splitext(file)[0]}_rotate.{os.path.splitext(file)[1]}"
            imsave(save_rotated_img_fp, np.uint8(synthesizer.synthesize_image(img_fp)))
            save_npy_fp = os.path.splitext(save_img_fp)[0]+".npy"

            file_list.write(f"{save_img_fp} {save_rotated_img_fp} {save_npy_fp}\n")
    file_list.close()


