from operator import pos
import os
import shutil
# switch to "osmesa" or "egl" before loading pyrender
os.environ["PYOPENGL_PLATFORM"] = "egl"

import dlib
import numpy as np
import math
import sys
np.set_printoptions(threshold=sys.maxsize)
import pyrender
import trimesh
import glob
from image_synthesis.model_fitting.model_fitting import load_3ddfa, get_param
from image_synthesis.model_fitting.utils.estimate_pose import angle2matrix
from skimage.io import imsave
import argparse
from math import sqrt

def get_rotation_matrix(rotation_directions):
    rotation_matrix = angle2matrix(rotation_directions)
    col4 = np.array([
        [0.0],
        [0.0],
        [0.0]
    ], dtype=np.float32)
    rotation_matrix = np.hstack((rotation_matrix, col4))
    row4 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    rotation_matrix = np.vstack((rotation_matrix, row4))
    return rotation_matrix

def elementwise_addition(first, second):
    return (first[0]+second[0], first[1]+second[1], first[2]+second[2])

def scale_triplet(triplet, scalar):
    return (triplet[0]*scalar[0], triplet[1]*scalar[1], triplet[2]*scalar[2])

def negate_pose_list(pose_list):
    return (-pose_list[0], -pose_list[1], -pose_list[2])

def negate_yaw_pose(pose_list):
    return (-pose_list[0], pose_list[1], pose_list[2])

def add_to_yaw(pose_list, n):
    return (pose_list[0]+n, pose_list[1], pose_list[2])

def add_to_pitch(pose_list, n):
    return (pose_list[0], pose_list[1]+n, pose_list[2])

def calculate_score(triplet):
    score = [abs(triplet[0]), abs(triplet[1]), abs(triplet[2])]
    scale = [10.0 if abs(pose)>0.2 else math.ceil(50.0*abs(pose)) for pose in triplet]
    scale = list(map(lambda x: (x+(x%2))/10.0, scale))
    return score, scale

def rotate_mesh(mesh_fp, save_fp, rotation_direction, yfov, return_cam_height=False):
        with open(mesh_fp) as f:
                mesh_lines = f.readlines()
        aligned_mesh_file = open(save_fp, "w")
        xmax = 0
        xmin = 0
        ymax = 0
        ymin = 0
        #zmax = 0
        #zmin = 0
        z_list = np.array([])
        y_acc = 0
        n_vertex = 0
        #cam_height = 0
        rotation_matrix = angle2matrix(rotation_direction)
        for line in mesh_lines:
            if line.startswith("v "):
                n_vertex += 1
                vertex = list(map(float, line.split(" ")[1:]))
                rotated_vertex = rotation_matrix.dot(np.array(vertex))
                y_acc += rotated_vertex[1]
                xmax = rotated_vertex[0] if rotated_vertex[0] > xmax else xmax
                xmin = rotated_vertex[0] if rotated_vertex[0] < xmin else xmin
                ymax = rotated_vertex[1] if rotated_vertex[1] > ymax else ymax
                ymin = rotated_vertex[1] if rotated_vertex[1] < ymin else ymin
                z_list = np.insert(z_list, z_list.searchsorted(rotated_vertex[2]), rotated_vertex[2])
                #zmax = rotated_vertex[2] if rotated_vertex[2] > zmax else zmax
                #zmin = rotated_vertex[2] if rotated_vertex[2] < zmin else zmin
                """
                if return_cam_height:
                    cam_height = rotated_vertex[1] if zmax==rotated_vertex[2] else cam_height
                """
                aligned_mesh_file.write(f"v {' '.join(map(str, rotated_vertex))}\n")
            else:
                aligned_mesh_file.write(line)
        aligned_mesh_file.close()
        height = (ymax-ymin)
        z_offset = z_list[round(n_vertex*0.85)]
        distance = (height/(2*np.tan(yfov)))
        xmid_point = xmin+((xmax-xmin)/2.0)
        """
        if not return_cam_height:
            cam_height = ymin+(height/2.0)
        """
        
        return (distance, z_offset), xmid_point, ymin+(height/2.0)

parser = argparse.ArgumentParser(description='Generate png from micc mesh.')
parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
parser.add_argument('--bbox_init', default='two', type=str,
                            help='one|two: one-step bbox initialization or two-step')
parser.add_argument('--path_prefix', default='/mnt/lhome/lhome/yongbk/florence', type=str, help='location of folder containing the meshes')

args = parser.parse_args()
model, alignment_model = load_3ddfa(args)

class Subject:
    def __init__(self, render, yfov, subject_number):
        subject_number_str = f"0{str(subject_number)}" if subject_number<10 else str(subject_number)
        self.mesh_directory_path = f"{args.path_prefix}/subject_{subject_number_str}/Model/frontal1/obj"
        mesh_fp = [fn for fn in glob.glob(f"{self.mesh_directory_path}/*.obj") 
            if not os.path.basename(fn).startswith('aligned') and
            not os.path.basename(fn).startswith('rotated')]
        print(mesh_fp)
        self.render = render
        eval_img_dir = f"{self.mesh_directory_path}/eval_img"
        if os.path.exists(eval_img_dir):
            shutil.rmtree(eval_img_dir)
        os.mkdir(eval_img_dir)
        self.yaw_step_size = np.radians(20)
        self.pitch_step_size = np.radians(15)
        self.yfov = yfov
        self.save_fp_prefix = f"{eval_img_dir}/subject_{subject_number_str}"


        score = [1.0, 1.0, 1.0]
        scale = [1.0, 1.0, 1.0]
        first_iteration = True
        self.aligned_mesh_fp = f"{self.mesh_directory_path}/aligned.obj"
        current_pose = (0.0, 0.0, 0.0)
        while not all(x <= 0.01 for x in score):
            if first_iteration:
                distance, xmid_point, self.cam_height = rotate_mesh(mesh_fp[0], self.aligned_mesh_fp, negate_pose_list(current_pose), yfov, False)
                first_iteration = False
            else:
                distance, xmid_point, self.cam_height = rotate_mesh(self.aligned_mesh_fp, self.aligned_mesh_fp, negate_pose_list(scale_triplet(current_pose, scale)), yfov, True)
            self.render.change_mesh(self.aligned_mesh_fp)
            self.render.change_camera_pose(xmid_point, self.cam_height, distance)
            current_pose = self.render(f"{self.save_fp_prefix}_0_0.jpg", False)
            score, scale = calculate_score(current_pose)
            print(current_pose)
            print(scale)
        self.render(f"{self.save_fp_prefix}_0_0.jpg")
        print("align done")
 
    
    def render_subject(self, pitch_step, yaw_step):
        yaw_target = yaw_step*self.yaw_step_size
        pitch_target = pitch_step*self.pitch_step_size
        rotated_mesh_fp = f"{self.mesh_directory_path}/rotated.obj"
        save_fp = f"{self.save_fp_prefix}_{pitch_step}_{yaw_step}.jpg"
        current_rotation_direction = (
            yaw_target,
            pitch_target,
            0.0
        )
        distance, xmid_point, self.cam_height = rotate_mesh(self.aligned_mesh_fp, rotated_mesh_fp, current_rotation_direction, self.yfov)
        self.render.change_mesh(rotated_mesh_fp)
        self.render.change_camera_pose(xmid_point, self.cam_height, distance)
        self.render(save_fp)

        return save_fp, current_rotation_direction


class Render:
    def __init__(self, yfov):
        self.r = pyrender.OffscreenRenderer(256, 256)
        self.camera_node = pyrender.Node(camera=pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0))
        light = pyrender.DirectionalLight(color=[1,1,1], intensity=12e3)
        self.mesh_node = None
        self.scene = pyrender.Scene(bg_color=(0.0, 0.0, 0.0))
        face_detector_path = 'Data/net-data/mmod_human_face_detector.dat'
        self.face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)

        self.scene.add_node(self.camera_node)
        self.scene.add(light, pose=np.eye(4))
    
    def change_mesh(self, mesh_fp):
        if self.mesh_node != None:
            self.scene.remove_node(self.mesh_node)
        self.mesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(trimesh.load(mesh_fp)))
        self.scene.add_node(self.mesh_node)
    
    def change_camera_pose(self, x, y, z):
        camera_pose = np.array([
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, 3.0*z[0]+z[1]],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.scene.set_pose(self.camera_node, camera_pose)


    def __call__(self, save_fp, check=True):
        #self.scene.set_pose(self.mesh_node, get_rotation_matrix(rotation_directions))
        color, _ = self.r.render(self.scene)
        detected_faces = self.face_detector(color, 1)
        if check:
            if len(detected_faces)==0:
                print(f"NO FACES {save_fp}")
        imsave(save_fp, color)
        _, _, _, current_pose = get_param(model, alignment_model, save_fp, args)
        return current_pose



if __name__ == '__main__':
    render = Render(np.pi / 3.0)
    faultyOBJs = [3, 27, 30]
    for i in range(27, 54):
        if i in faultyOBJs:
            continue
        else:
            print(i)
            subject = Subject(render, np.pi / 3.0, i)
            transformation_file = open(f"{subject.mesh_directory_path}/transformations.txt", "w")
            for pitch_step in range(-1, 2):
                for yaw_step in range(-4, 5):
                    save_fp, rotation_from_init = subject.render_subject(pitch_step, yaw_step)
                    transformation_file.write(f"{save_fp} {' '.join(map(str, rotation_from_init))}\n")
            transformation_file.close()