from math import degrees
from scipy.io import loadmat

#each dataset entry is named as such: "{dataset_name}_{image_id}_{face_id}_{pose_number}"
#example: HELEN_2667636661_3_15
#we define face_file_path to be {dataset_name}_{image_id}_{face_id}
class Face:
    def __init__(self, face_file_path):
        self.face_file_path = face_file_path
        self.face_poses = []

    def add_face_pose(self, pose_number):
        self.face_poses.append(FacePose(self.face_file_path, pose_number))


class FacePose:
    def __init__(self, face_file_path, pose_number):
        file_path = "{}_{}".format(face_file_path, pose_number)
        self.img_path = file_path + ".jpg"
        self.MM_param_path = file_path + ".mat"
        self.pose = degrees(loadmat(self.MM_param_path)['Pose_Para'][0][1])
        