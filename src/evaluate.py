import argparse
import glob
import os

import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KDTree

from face_reconstruction import Face_reconstructor
#from PRNet.reconstruct import PRNet
from image_synthesis.model_fitting.utils.estimate_pose import angle2matrix

from icp.icp import icp
from Utils.write import write_obj_with_colors


def apply_homogenous_tform(tform, vertices):
    n, m = vertices.shape
    vertices_affine = np.ones((n, m + 1))
    vertices_affine[:, :3] = vertices.copy()
    vertices = np.dot(tform, vertices_affine.T).T
    return vertices[:, :3]

def get_rotation_matrix(pose_list):
    rotation_matrix = angle2matrix(pose_list)
    col4 = np.array([
        [0.0],
        [0.0],
        [0.0]
    ], dtype=np.float32)
    rotation_matrix = np.hstack((rotation_matrix, col4))
    row4 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    rotation_matrix = np.vstack((rotation_matrix, row4))
    return rotation_matrix


def get_vertices_from_obj(obj_path):
    with open(obj_path) as f:
        lines = f.readlines()
    vertices = [line for line in lines if line.startswith('v ')]
    vertices = strip_obj_string(vertices)
    return vertices


def strip_obj_string(lines):
    array = np.zeros((len(lines), 3))
    for i, line in enumerate(lines):
        sub_array = np.array(line[2:].split(' ')).astype(np.float32)[:3]
        array[i] = sub_array
    return array


class prediction_evaluater:
    def __init__(self):
        self.face_ind = np.loadtxt('Data/uv-data/face_ind.txt').astype(np.int32)
        self.uv_kpt_ind = np.loadtxt('Data/uv-data/uv_kpt_ind.txt').astype(np.int32)
        self.triangles = np.loadtxt('Data/uv-data/triangles.txt').astype(np.int32)

    def __call__(self, predicted_vertices, ground_truth_vertices, init_pose=None, save_vertices=False,
                 save_output='aligned_vertices.obj'):
        """
        if alignment_data is not None:
            init_pose = alignment_data[:4]
            scale = alignment_data[4][0]
        else:
            init_pose = None
            scale = 1.0
        """
        scale = 1.0
        original_predicted_vertices = predicted_vertices.copy() * scale
        original_f_vertices = ground_truth_vertices.copy()
        if (predicted_vertices.shape[0] > ground_truth_vertices.shape[0]):
            diff = predicted_vertices.shape[0] - ground_truth_vertices.shape[0]
            predicted_vertices = predicted_vertices[diff:, :] * scale
        else:
            diff = ground_truth_vertices.shape[0] - predicted_vertices.shape[0]
            ground_truth_vertices = ground_truth_vertices[diff:, :] * scale

        tform, distances, i = icp(predicted_vertices, ground_truth_vertices,
                                  max_iterations=100, tolerance=0.0001, init_pose=init_pose)

        aligned_predicted_vertices = apply_homogenous_tform(tform, predicted_vertices)
        aligned_original_vertices = apply_homogenous_tform(tform, original_predicted_vertices)

        if save_vertices:
            colors = np.ones((aligned_original_vertices.shape))
            write_obj_with_colors(save_output, aligned_original_vertices, self.triangles, colors)

        error = self.nmse(aligned_original_vertices, original_f_vertices)
        return error

    def nmse(self, predicted_vertices, ground_truth_vertices, normalization_factor=None):
        # calculate the normalized mean squared error between a predicted and ground truth mesh
        if not normalization_factor:
            mins = np.amin(ground_truth_vertices, axis=0)
            maxes = np.amax(ground_truth_vertices, axis=0)
            bbox = np.sqrt((maxes[0] - mins[0]) ** 2 + (maxes[1] - mins[1]) ** 2 + (maxes[2] - mins[2]) ** 2)
            normalization_factor = bbox

        v_tree = KDTree(ground_truth_vertices)
        error_array = np.zeros(predicted_vertices.shape[0])
        for i, v in enumerate(predicted_vertices):
            dst, ind = v_tree.query([v], k=1)
            gt_v = ground_truth_vertices[ind[0][0]]
            error_array[i] = distance.euclidean(v, gt_v)

        nmse = np.mean(error_array) / normalization_factor
        print(nmse)
        return nmse


def evaluate_predictions(args):
    evaluation_dir = "./evaluation_cropped"
    predicted_path = f"{evaluation_dir}/predicted.obj"
    evaluater = prediction_evaluater()
    reconstuctor = Face_reconstructor()
    #reconstuctor = PRNet()
    faultyOBJs = [3, 30]
    for i in range(1, 54):
        if i in faultyOBJs:
            continue
        else:
            print(i)
            subject_number_str = f"0{str(i)}" if i<10 else str(i)
            mesh_path = f"{args.florence_prefix}/subject_{subject_number_str}/Model/frontal1/obj"
            gt_vertices = get_vertices_from_obj(f"{mesh_path}/aligned.obj")
            f = open(f"{mesh_path}/transformations.txt", "r")
            error_file = open(f"{evaluation_dir}/{subject_number_str}.txt", "w")
            for line in f:
                image_info = line.strip().split(" ")
                alignment_matrix = get_rotation_matrix([-float(pose) for pose in image_info[1:]])
                if(reconstuctor.reconstruct(image_info[0], predicted_path)):
                    predicted_vertices = get_vertices_from_obj(predicted_path)
                    error = evaluater(predicted_vertices, gt_vertices, init_pose=alignment_matrix)
                    error_file.write(f"{image_info[0]} {error}\n")
                else:
                    error_file.write(f"{image_info[0]} no_detected_face\n")
            f.close()
            error_file.close()
    return


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Network Evaluation')
    par.add_argument('--florence_prefix', default='/lhome/yongbk/florence', type=str,
                     help='The path to the florence dataset description file')
    evaluate_predictions(par.parse_args())

