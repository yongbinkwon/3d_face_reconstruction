import os
import glob

def get_faces_from_dataset(root_folder):
    for dataset in os.listdir(root_folder):
        for file in glob.glob(os.path.join(root_folder, dataset, "*.jpg")):
            print(file)

get_faces_from_dataset(os.path.join("src", "300W-LP"))