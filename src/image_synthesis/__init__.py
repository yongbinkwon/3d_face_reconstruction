import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
import skimage.transform as trans
from .options.test_options import TestOptions
from .models.test_model import TestModel
from .util import util
from .models.networks.rotate_render import TestRender
from .data.data_utils import get_input
from .model_fitting.model_fitting import load_3ddfa, get_param


def rotate_yaw_pose(yaw_pose):
    abs_yaw_pose = abs(yaw_pose)
    if abs_yaw_pose <= np.pi/4:
        abs_yaw_pose += np.pi/4
    elif abs_yaw_pose <= np.pi/2:
        abs_yaw_pose -= np.pi/4
    else:
        print("too large pose")
        return yaw_pose
    return np.sign(yaw_pose)*abs_yaw_pose


class Synthesize():

    def __init__(self):
        self.opt = TestOptions().parse()
        self.ngpus = self.opt.device_count

        self.render_gpu_ids = list(range(self.ngpus - self.opt.render_thread, self.ngpus))
        self.render_layer_list = []
        for gpu in self.render_gpu_ids:
            self.opt.gpu_ids = gpu
            render_layer = TestRender(self.opt)
            self.render_layer_list.append(render_layer)

        self.opt.gpu_ids =  list (range (0, self.ngpus))
        print('Testing gpu ', self.opt.gpu_ids)

        self.model = TestModel(self.opt)
        self.model.eval()
        self.model = torch.nn.DataParallel(self.model.cuda(),
                                    device_ids=self.opt.gpu_ids,
                                    output_device=self.opt.gpu_ids[-1],
                                    )
        self.save_path = os.path.join(self.opt.save_path, self.opt.dataset_name)

        # test
        self.fitting_model, self.alignment_model = load_3ddfa(self.opt)


    def affine_align(self, img, landmark=None, **kwargs):
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

    def synthesize_image(self, img_fp_suffix):
        img_fp = os.path.join(self.opt.img_prefix, img_fp_suffix)
        param, landmarks, img_orig, yaw_pose = get_param(
            self.fitting_model, self.alignment_model, img_fp, self.opt
        )
        rotated_yaw_pose = rotate_yaw_pose(yaw_pose)
        landmarks = np.array(landmarks).reshape(5, 2)
        if img_orig is None:
            raise Exception('No Image')
        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        wrapped_img, M = self.affine_align(img, landmarks)
        M = torch.from_numpy(np.expand_dims(M, axis=0)).float()
        wrapped_img = wrapped_img.transpose(2, 0, 1) / 255.0
        wrapped_img = torch.from_numpy(np.expand_dims(wrapped_img, axis=0)).float()
        data = get_input(wrapped_img, M, self.render_layer_list[0], rotated_yaw_pose, param)

        rotated_image = self.model.forward(data, mode='single')
        return_image = util.tensor2im(rotated_image[0])

        if self.opt.save_image:
            rotated_image_savepath = os.path.join(self.save_path, img_fp_suffix)
            util.save_image(return_image, rotated_image_savepath, create_dir=True)
        
        return return_image




