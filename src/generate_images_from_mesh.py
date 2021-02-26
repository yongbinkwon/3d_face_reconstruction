import os
# switch to "osmesa" or "egl" before loading pyrender
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pyrender
import trimesh
import glob
from image_synthesis.model_fitting.model_fitting import load_3ddfa, get_param
from image_synthesis.model_fitting.utils.estimate_pose import angle2matrix
from skimage.io import imsave
import argparse
from math import sqrt

def get_rotation_matrix(pose_list):
    rotation_matrix = angle2matrix(negate_pose_list(pose_list))
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

def negate_pose_list(pose_list):
    return (-pose_list[0], -pose_list[1], -pose_list[2])

def negate_yaw_pose(pose_list):
    return (-pose_list[0], pose_list[1], pose_list[2])

def add_to_yaw(pose_list, n):
    return (pose_list[0]+n, pose_list[1], pose_list[2])

def calculate_score(triplet):
    return sqrt((triplet[0]**2)+(triplet[0]**2)+(triplet[0]**2))

def generate_mesh_image(args, subject_number):
    subject_number_str = f"0{str(subject_number)}" if subject_number<10 else str(subject_number)
    model, alignment_model = load_3ddfa(args)
    mesh_directory_path = f"{args.path_prefix}/subject_{subject_number_str}/Model/frontal1/obj"
    mesh_names = glob.glob(f"{mesh_directory_path}/*.obj")
    if len(mesh_names) > 1:
        raise ValueError("should only be one .obj file")
    mesh_fp = mesh_names[0]
    micc_mesh = trimesh.load(mesh_fp)
    mesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(micc_mesh))
    scene = pyrender.Scene()
    scene.add_node(mesh_node)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 425.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)
    scene.add(light, pose=np.eye(4))
    r = pyrender.OffscreenRenderer(256, 256)
    score = 1
    final_poses = (0.0, 0.0, 0.0)

    eval_img_dir = f"{mesh_directory_path}/eval_img"
    if not os.path.exists(eval_img_dir):
        os.mkdir(eval_img_dir)

    while(score > 0.001):
        scene.set_pose(mesh_node, get_rotation_matrix(final_poses))
        color, _ = r.render(scene)
        imsave(f"{eval_img_dir}/subject_{subject_number_str}_0.png", color)
        _, _, _, poses = get_param(model, alignment_model, f"{eval_img_dir}/subject_{subject_number_str}_0.png", args)
        final_poses = elementwise_addition(final_poses, poses)
        score = calculate_score(poses)
        print(score)
    
    step_size = np.pi/8
    for step in range(1, 5):
        final_poses = add_to_yaw(final_poses, step_size)

        scene.set_pose(mesh_node, get_rotation_matrix(final_poses))
        color, _ = r.render(scene)
        imsave(f"{eval_img_dir}/subject_{subject_number_str}_{str(step)}.png", color)
        if step==4:
            _, _, _, poses = get_param(model, alignment_model, f"{eval_img_dir}/subject_{subject_number_str}_{str(step)}.png", args)
            diff = np.pi/2 - poses[0] 
            while diff < 0:
                final_poses = add_to_yaw(final_poses, diff)
                scene.set_pose(mesh_node, get_rotation_matrix(final_poses))
                color, _ = r.render(scene)
                imsave(f"{eval_img_dir}/subject_{subject_number_str}_{str(step)}.png", color)
                _, _, _, poses = get_param(model, alignment_model, f"{eval_img_dir}/subject_{subject_number_str}_{str(step)}.png", args)
                diff = np.pi/2 - poses[0] 
                


        scene.set_pose(mesh_node, get_rotation_matrix(negate_yaw_pose(final_poses)))
        color, _ = r.render(scene)
        imsave(f"{eval_img_dir}/subject_{subject_number_str}_-{str(step)}.png", color)
        if step==4:
            _, _, _, poses = get_param(model, alignment_model, f"{eval_img_dir}/subject_{subject_number_str}_-{str(step)}.png", args)
            diff = np.pi/2 + poses[0] 
            while diff < 0:
                final_poses = add_to_yaw(final_poses, diff)
                scene.set_pose(mesh_node, get_rotation_matrix(negate_yaw_pose(final_poses)))
                color, _ = r.render(scene)
                imsave(f"{eval_img_dir}/subject_{subject_number_str}_-{str(step)}.png", color)
                _, _, _, poses = get_param(model, alignment_model, f"{eval_img_dir}/subject_{subject_number_str}_-{str(step)}.png", args)
                diff = np.pi/2 - poses[0] 



    

parser = argparse.ArgumentParser(description='Generate png from micc mesh.')
parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
parser.add_argument('--bbox_init', default='two', type=str,
                            help='one|two: one-step bbox initialization or two-step')
parser.add_argument('--path_prefix', default='/lhome/yongbk/florence', type=str, help='location of folder containing the meshes')

args = parser.parse_args()

generate_mesh_image(args, 35)

