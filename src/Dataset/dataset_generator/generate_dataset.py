import os
from skimage import io
import skimage.transform
import numpy as np
from image_synthesis import Synthesize
import face3d
from face3d.morphable_model import MorphabelModel
from .pos_map_code import process_uv, run_posmap_300W_LP
from Dataset.dataset_generator.facegen_to_posmap import generate_posmap_facegen_bfm
from tqdm import tqdm

"""
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
"""

def get_random_subfolder(folder_path, is_image=False):
    dirs = np.array(os.listdir(folder_path))
    if is_image:
        valid_dirs_index = [i for i, item in enumerate(dirs) if 'jpg' in item or 'png' in item]
        dirs = dirs[valid_dirs_index]
    random_dir = np.random.choice(dirs)
    random_dir_path = os.path.join(folder_path, random_dir)
    return random_dir_path

def apply_random_background(image):
    dtd_path = 'Data/dtd/images'  # texture dataset
    random_dir_path = get_random_subfolder(dtd_path)  # random category
    random_img_path = get_random_subfolder(random_dir_path, is_image=True)  # random image within category

    bg_img = io.imread(random_img_path)
    if (bg_img.shape[0] < 256 or bg_img.shape[
        1] < 256):  # should not happend according to texture dataset specifications
        return image  # return image without background image if this is the case
    image_with_bg = image[:, :, :3].copy()
    cropped_bg_img = bg_img[:256, :256, :3].copy()  # only use the top left 256x256 pixels
    background_mask = np.array(image_with_bg <= [0, 0, 0])  # if rgb values are 0, background should be shown
    image_with_bg[background_mask] = cropped_bg_img[background_mask]

    return image_with_bg


def generate_facegen_dataset(dataset_root_folder, save_root_folder, file_list):
    bfm = MorphabelModel('Data/BFM/BFM.mat')
    uv_coords = face3d.morphable_model.load.load_uv_coords('Data/BFM/BFM_UV.mat')
    uv_coords = process_uv(uv_coords)

    synthesizer = Synthesize()
    image_h, image_w = 256, 256

    already_generated_subjects = os.listdir(save_root_folder)
    subjects_to_generate = list(filter(lambda directory: directory not in already_generated_subjects, os.listdir(dataset_root_folder)))
    #subjects_to_generate = os.listdir(dataset_root_folder)
    with open(file_list, 'a') as f:
        for subject_number in subjects_to_generate:
            print(f"subject {subject_number}")
            obj_fp = os.path.join(dataset_root_folder, subject_number, f"subject_{subject_number}.obj")
            save_folder = os.path.join(save_root_folder, subject_number)
            for i in range(-4, 4):
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                current_img_fp = os.path.join(dataset_root_folder, subject_number, f"render_{i}.png")
                img_save_fp = os.path.join(save_folder, f"orig_{i}.png")
                rotated_img_fp = os.path.join(save_folder, f"rotated_{i}.png")
                posmap_fp = os.path.join(save_folder, f"posmap_{i}.npy")
                cropping_tform = generate_posmap_facegen_bfm(bfm, uv_coords, current_img_fp, obj_fp, posmap_fp, save_image=True)
                current_img = io.imread(current_img_fp)
                current_img_cropped = skimage.transform.warp(current_img, cropping_tform.inverse,
                                                            output_shape=(image_h, image_w), preserve_range=True)
                img = apply_random_background(current_img_cropped.astype(np.uint8))
                io.imsave(img_save_fp, img, check_contrast=False)
                
                try:
                    synthesizer.synthesize_image(img_save_fp, rotated_img_fp)
                except TypeError:
                    print("bricked"+img_save_fp)
                    continue
                

                f.write(img_save_fp + ' ' + rotated_img_fp + ' ' + posmap_fp + '\n')

def sort_dataset(dataset_root_folder, save_root_folder, file_list):
    bfm = MorphabelModel('Data/BFM/BFM.mat')
    uv_coords = face3d.morphable_model.load.load_uv_coords('Data/BFM/BFM_UV.mat')
    uv_coords = process_uv(uv_coords)

    synthesizer = Synthesize()
    image_h, image_w = 256, 256

    with open(file_list, 'w') as f:
        for i in range(1, 2407):
            subject_number = str(i)
            print(f"subject {subject_number}")
            obj_fp = os.path.join(dataset_root_folder, subject_number, f"subject_{subject_number}.obj")
            save_folder = os.path.join(save_root_folder, subject_number)
            for i in range(-4, 4):
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                current_img_fp = os.path.join(dataset_root_folder, subject_number, f"render_{i}.png")
                img_save_fp = os.path.join(save_folder, f"orig_{i}.png")
                rotated_img_fp = os.path.join(save_folder, f"rotated_{i}.png")
                posmap_fp = os.path.join(save_folder, f"posmap_{i}.npy")
                if not os.path.isfile(img_save_fp) or not os.path.isfile(posmap_fp):
                    cropping_tform = generate_posmap_facegen_bfm(bfm, uv_coords, current_img_fp, obj_fp, posmap_fp, save_image=True)
                    current_img = io.imread(current_img_fp)
                    current_img_cropped = skimage.transform.warp(current_img, cropping_tform.inverse,
                                                            output_shape=(image_h, image_w), preserve_range=True)
                    img = apply_random_background(current_img_cropped.astype(np.uint8))
                    io.imsave(img_save_fp, img, check_contrast=False)
                
                if not os.path.isfile(rotated_img_fp):
                    try:
                        synthesizer.synthesize_image(img_save_fp, rotated_img_fp)
                    except TypeError:
                        print("bricked"+img_save_fp)
                        continue

                f.write(img_save_fp + ' ' + rotated_img_fp + ' ' + posmap_fp + '\n')

def generate_rest(dataset_root_folder, save_root_folder, file_list):
    bfm = MorphabelModel('Data/BFM/BFM.mat')
    uv_coords = face3d.morphable_model.load.load_uv_coords('Data/BFM/BFM_UV.mat')
    uv_coords = process_uv(uv_coords)

    image_h, image_w = 256, 256

    with open(file_list, 'a') as f:
        for subject_number in os.listdir(dataset_root_folder):
            num_files = len(os.listdir(os.path.join(save_root_folder, subject_number)))
            if (num_files != 32):
                print(num_files)
                print(f"subject {subject_number}")
                obj_fp = os.path.join(dataset_root_folder, subject_number, f"subject_{subject_number}.obj")
                save_folder = os.path.join(save_root_folder, subject_number)
                for i in range(-4, 4):
                    img_save_fp = os.path.join(save_folder, f"orig_{i}.png")
                    posmap_fp = os.path.join(save_folder, f"posmap_{i}.npy")
                    rotated_img_fp = os.path.join(save_folder, f"rotated_{i}.png")
                    if(os.path.isfile(img_save_fp) and os.path.isfile(posmap_fp) and os.path.isfile(rotated_img_fp)):
                        continue
                    current_img_fp = os.path.join(dataset_root_folder, subject_number, f"render_{i}.png")
                    cropping_tform = generate_posmap_facegen_bfm(bfm, uv_coords, current_img_fp, obj_fp, posmap_fp, save_image=True)
                    current_img = io.imread(current_img_fp)
                    current_img_cropped = skimage.transform.warp(current_img, cropping_tform.inverse,
                                                                output_shape=(image_h, image_w), preserve_range=True)
                    img = apply_random_background(current_img_cropped.astype(np.uint8))
                    io.imsave(img_save_fp, img, check_contrast=False)

                    print(img_save_fp + ' ' + rotated_img_fp + ' ' + posmap_fp)
                    f.write(img_save_fp + ' ' + rotated_img_fp + ' ' + posmap_fp + '\n')

def fix_dataset(dataset_root_folder, save_root_folder, file_list):
    bfm = MorphabelModel('Data/BFM/BFM.mat')
    uv_coords = face3d.morphable_model.load.load_uv_coords('Data/BFM/BFM_UV.mat')
    uv_coords = process_uv(uv_coords)

    image_h, image_w = 256, 256

    with open(file_list, 'a') as f:
        for subject_number in os.listdir(dataset_root_folder):
            obj_fp = os.path.join(dataset_root_folder, subject_number, f"subject_{subject_number}.obj")
            save_folder = os.path.join(save_root_folder, subject_number)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            for i in range(-4, 4):
                img_save_fp = os.path.join(save_folder, f"orig_{i}.png")
                posmap_fp = os.path.join(save_folder, f"posmap_{i}.npy")
                rotated_img_fp = os.path.join(save_folder, f"rotated_{i}.png")
                if(os.path.isfile(img_save_fp) and os.path.isfile(posmap_fp)):
                    f.write(img_save_fp + ' ' + rotated_img_fp + ' ' + posmap_fp + '\n')
                else:
                    current_img_fp = os.path.join(dataset_root_folder, subject_number, f"render_{i}.png")
                    cropping_tform = generate_posmap_facegen_bfm(bfm, uv_coords, current_img_fp, obj_fp, posmap_fp, save_image=True)
                    current_img = io.imread(current_img_fp)
                    current_img_cropped = skimage.transform.warp(current_img, cropping_tform.inverse,
                                                                output_shape=(image_h, image_w), preserve_range=True)
                    img = apply_random_background(current_img_cropped.astype(np.uint8))
                    io.imsave(img_save_fp, img, check_contrast=False)

                    print(img_save_fp + ' ' + rotated_img_fp + ' ' + posmap_fp)
                    f.write(img_save_fp + ' ' + rotated_img_fp + ' ' + posmap_fp + '\n')

def generate_rotated(file_list):
    synthesizer = Synthesize()
    with open(file_list, 'r') as f:
        rotate_filelist = open(os.path.join(os.path.dirname(file_list), "rotate_filelist.txt"), 'w+')
        for line in tqdm(f):
            img_fp, save_fp, _ = line.strip().split(' ')
            img_fp = "/".join(["/disk2"]+img_fp.split("/")[2:])
            save_fp = "/".join(["/disk2"]+save_fp.split("/")[2:])
            try:
                synthesizer.synthesize_image(img_fp, save_fp)
                rotate_filelist.write(line)
            except TypeError:
                print("bricked "+line)
                continue
            """
            print(img_fp)
            synthesizer.synthesize_image(img_fp, save_fp)
            rotate_filelist.write(line)
            """
            
        rotate_filelist.close()
