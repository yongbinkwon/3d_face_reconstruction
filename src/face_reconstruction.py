from image_synthesis.util.util import save_image
import os

import cv2
import dlib
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rescale, estimate_transform, warp

from Networks.predictor import MobilenetPosPredictor
from Utils.write import write_obj_with_colors
from image_synthesis import Synthesize
from tqdm import tqdm

#import tensorflow as tf

def mask_pos(pos):
    '''
    remove neck/irrelevant regions
    '''
    mask_path = 'Data/uv-data/facegen_face_mask.png'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    index_mask = mask[:, :] < 0.5
    masked_pos = pos.copy()
    masked_pos[index_mask] = [0, 0, 0]
    return masked_pos


def plot_vertices_on_image_from_pos(pos, l68, front_img):
    h, w, c = pos.shape
    plotted_front_img = front_img.copy().astype(np.uint8)
    h_i, w_i, c_i = plotted_front_img.shape
    max_h = np.max(pos[:, :, 1])
    max_w = np.max(pos[:, :, 0])
    min_z = int(np.min(pos[:, :, 2]))
    max_z = int(np.max(pos[:, :, 2]))

    if (max_w - w_i) > 0:
        enlarged_front_img = np.zeros((h_i, int(max_w), 3), dtype=np.uint8)
        enlarged_front_img[:, 0:w_i, :] = plotted_front_img[:, :, :]
        plotted_front_img = enlarged_front_img
        h_i, w_i, c_i = plotted_front_img.shape

    if (max_h - h_i) > 0:
        enlarged_front_img = np.zeros((int(max_h), w_i, 3), dtype=np.uint8)
        enlarged_front_img[0:h_i, :, :] = plotted_front_img[:, :, :]
        plotted_front_img = enlarged_front_img
        h_i, w_i, c_i = plotted_front_img.shape

    for h_u in range(h):
        for w_u in range(w):
            index = np.around(pos[h_u][w_u], decimals=1).astype(int)
            plotted_front_img[index[1] - 2][index[0] - 2] = [0, 255 - (max_z - index[2]), index[2]]

    for (x, y) in l68:
        plotted_front_img[y][x] = [255, 0, 0]  # draw red at coord
        plotted_front_img[y + 1][x + 1] = [255, 0, 0]  # draw cross to show keypoints more clearly
        plotted_front_img[y + 1][x - 1] = [255, 0, 0]
        plotted_front_img[y - 1][x + 1] = [255, 0, 0]
        plotted_front_img[y - 1][x - 1] = [255, 0, 0]
    return plotted_front_img


# from PRNet code
def get_cropping_transformation(image, face_detector, shape_predictor):
    detected_faces = face_detector(image, 1)
    if len(detected_faces) == 0:
        print('warning: no detected face')
        return

    d = detected_faces[
        0].rect  ## only use the first detected face (assume that each input image only contains one face)
    left = d.left()
    right = d.right()
    top = d.top()
    bottom = d.bottom()
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.14])
    size = int(old_size * 1.58)

    
    shape = shape_predictor(image, d)
    coords = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, 255], [255, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    return [coords, tform]


def uncrop_pos(cropped_pos, cropping_tform):
    cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
    z = cropped_vertices[2, :].copy() / cropping_tform.params[0, 0]
    cropped_vertices[2, :] = 1
    vertices = np.dot(np.linalg.inv(cropping_tform.params), cropped_vertices)
    vertices = np.vstack((vertices[:2, :], z))
    pos = np.reshape(vertices.T, [256, 256, 3])
    return pos


def get_cropped_image(img, cropping_tform):
    float_img = img / 256.0 / 1.1
    if not cropping_tform:
        return float_img
    else:
        return warp(float_img, cropping_tform.inverse, output_shape=(256, 256))


class Face_reconstructor:
    def __init__(self):
        self.model_path = 'Data/net-data/trained_fg_then_real.h5'  # trained_fg_then_real.h5'
        self.face_detector_path = 'Data/net-data/mmod_human_face_detector.dat'
        self.shape_predictor_path = 'Data/net-data/shape_predictor_68_face_landmarks.dat'
        self.img_type = '.jpg'

        self.synthesizer = Synthesize()

        self.triangles = np.loadtxt('Data/uv-data/triangles.txt').astype(np.int32)
        self.face_ind = np.loadtxt('Data/uv-data/face_ind.txt').astype(np.int32)
        self.face_detector = dlib.cnn_face_detection_model_v1(self.face_detector_path)
        self.shape_predictor = dlib.shape_predictor(self.shape_predictor_path)

        self.pos_predictor = MobilenetPosPredictor(256, 256)
        self.mobilenet_pos_predictor = os.path.join('', self.model_path)  # Data/net-data/keras_mobilenet_prn_20_epochs_097.h5')
        if not os.path.isfile(self.mobilenet_pos_predictor):
            print("please download trained model first.")
            exit()
        self.pos_predictor.restore(self.mobilenet_pos_predictor)

        self.save_images = False
        

    def reconstruct(self, img_fp, obj_save_fp):

        front_img = imread(img_fp)[:, :, :3]
        side_img = self.synthesizer.synthesize_image(img_fp)[:, :, :3]

        if front_img.shape != (256, 256, 3):
            max_size = max(front_img.shape[0], front_img.shape[1])
            if max_size > 1000:
                front_img = rescale(front_img, 1000. / max_size)
                front_img = (front_img * 255).astype(np.uint8)
            front_img = np.around(front_img, decimals=1).astype(np.uint8)

        if side_img.shape != (256, 256, 3):
            max_size = max(side_img.shape[0], side_img.shape[1])
            if max_size > 1000:
                side_img = rescale(side_img, 1000. / max_size)
                side_img = (side_img * 255).astype(np.uint8)
            side_img = np.around(side_img, decimals=1).astype(np.uint8)

        #l68_front, cropping_tform_front = get_cropping_transformation(front_img, self.face_detector, self.shape_predictor)
        #_, cropping_tform_side = get_cropping_transformation(side_img, self.face_detector, self.shape_predictor)
        cropping_tform_front = get_cropping_transformation(front_img, self.face_detector, self.shape_predictor)
        cropping_tform_side = get_cropping_transformation(side_img, self.face_detector, self.shape_predictor)
        if (cropping_tform_front is None or cropping_tform_side is None):
            return False

        cropped_image_front = get_cropped_image(front_img, cropping_tform_front[1])
        cropped_image_side = get_cropped_image(side_img, cropping_tform_side[1])
        
        if self.save_images:
            imsave(f"{os.path.dirname(obj_save_fp)}/front_cropped.jpg", cropped_image_front)
            imsave(f"{os.path.dirname(obj_save_fp)}/side_cropped.jpg", cropped_image_side)
        

        img_concat = np.concatenate((cropped_image_front, cropped_image_side), axis=2)
        cropped_pos = self.pos_predictor.predict(img_concat)

        pos = uncrop_pos(cropped_pos, cropping_tform_front[1])

        all_vertices = np.reshape(pos, [256 ** 2, -1])
        vertices = all_vertices[self.face_ind, :]

        save_vertices = vertices.copy()
        save_vertices[:, 1] = 256 - 1 - save_vertices[:, 1]

        [h, w, _] = front_img.shape
        vertices[:, 0] = np.minimum(np.maximum(vertices[:, 0], 0), w - 1)  # x
        vertices[:, 1] = np.minimum(np.maximum(vertices[:, 1], 0), h - 1)  # y
        ind = np.round(vertices).astype(np.int32)
        colors = front_img[ind[:, 1], ind[:, 0], :]  # n x 3

        
        write_obj_with_colors(obj_save_fp, save_vertices, self.triangles, colors)

        if self.save_images:
            masked_pos = mask_pos(pos)
            plotted_image = plot_vertices_on_image_from_pos(masked_pos, cropping_tform_front[0], front_img)
            imsave(f"{os.path.dirname(obj_save_fp)}/projected.jpg", plotted_image)

        return True

if __name__=='__main__':
    reconstructor = Face_reconstructor()
    reconstructor.reconstruct("/lhome/yongbk/florence/subject_02/Model/frontal1/obj/eval_img/subject_02_-1_-4.jpg", "./")
        
