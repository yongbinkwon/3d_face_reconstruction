import numpy as np

import skimage.transform as trans
import cv2

import os
import torch




def get_input(data, render, yaw_pose, param):
    real_image = data['image']
    rotated_meshs = []
    rotated_landmarks_list = []
    original_angles_list = []
    rotated_landmarks_list_106 = []
    paths = []
    real_images = []
    pose_list = []
    rotated_mesh, rotate_landmarks, original_angles, rotate_landmarks_106\
        = render.rotate_render(param, real_image, data['M'], yaw_pose=yaw_pose)
    rotated_meshs.append(rotated_mesh)
    rotated_landmarks_list.append(rotate_landmarks)
    rotated_landmarks_list_106.append(rotate_landmarks_106)
    original_angles_list.append(original_angles)
    paths += data['path']
    pose_list += ['yaw_{}'.format(yaw_pose) for i in range(len(data['path']))]
    real_images.append(real_image)
    rotated_meshs = torch.cat(rotated_meshs, 0)
    rotated_landmarks_list = torch.cat(rotated_landmarks_list, 0)
    rotated_landmarks_list_106 = torch.cat(rotated_landmarks_list_106, 0)
    original_angles_list = torch.cat(original_angles_list, 0)
    output = {}
    real_image = real_image * 2 - 1
    rotated_meshs = rotated_meshs * 2 - 1
    output['image'] = real_image.cpu()
    output['rotated_mesh'] = rotated_meshs.cpu()
    output['rotated_landmarks'] = rotated_landmarks_list.cpu()
    output['rotated_landmarks_106'] = rotated_landmarks_list_106.cpu()
    output['original_angles'] = original_angles_list.cpu()
    output['path'] = paths
    output['pose_list'] = pose_list
    return output


def affine_align(self, im, landmark=None, **kwargs):
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    M = None
    h, w, c = img.shape
    src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]], dtype=np.float32)
    src = src * 290 / 112
    src[:, 0] += 50
    src[:, 1] += 60
    src = src / 400 * self.opt.crop_size
    dst = landmark
    # dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]
    warped = cv2.warpAffine(img, M, (self.opt.crop_size, self.opt.crop_size), borderValue=0.0)
    return warped, M
