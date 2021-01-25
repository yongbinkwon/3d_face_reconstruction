import os
import numpy as np
from .options.test_options import TestOptions
from .models.test_model import TestModel
from .util import util
import torch
import cv2
import skimage.transform as trans
from .models.networks.rotate_render import TestRender
from tqdm import tqdm
from .data.data_utils import get_input
from .model_fitting.model_fitting import load_3ddfa, get_param


def create_path(a_path, b_path):
    name_id_path = os.path.join(a_path, b_path)
    if not os.path.exists(name_id_path):
        os.makedirs(name_id_path)
    return name_id_path

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


def create_paths(save_path, img_path, foldername='orig', folderlevel=2, pose='0'):
    save_rotated_path_name = create_path(save_path, foldername)

    path_split = img_path.split('/')
    rotated_file_savepath = save_rotated_path_name
    for level in range(len(path_split) - folderlevel, len(path_split)):
        file_name = path_split[level]
        if level == len(path_split) - 1:
            file_name = str(pose) + '_' + file_name
        rotated_file_savepath = os.path.join(rotated_file_savepath, file_name)
    return rotated_file_savepath


def save_img(img, save_path):
    image_numpy = util.tensor2im(img)
    util.save_image(image_numpy, save_path, create_dir=True)
    return image_numpy

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

class Synthesize():

    def __init__(self):
        
        self.opt = TestOptions().parse()

        self.data_info = data.dataset_info()
        self.datanum = self.data_info.get_dataset(self.opt)[0]
        self.folderlevel = self.data_info.folder_level[self.datanum]

        self.dataloaders = data.create_dataloader_test(self.opt)

        self.ngpus = self.opt.device_count

        self.render_gpu_ids = list(range(self.ngpus - self.opt.render_thread, self.ngpus))
        self.render_layer_list = []
        for gpu in self.render_gpu_ids:
            self.opt.gpu_ids = gpu
            render_layer = TestRender(self.opt)
            self.render_layer_list.append(render_layer)

        self.opt.gpu_ids =  list (range (0, self.ngpus))
        print('Testing gpu ', self.opt.gpu_ids)
        if self.opt.names is None:
            self.model = TestModel(self.opt)
            self.model.eval()
            self.model = torch.nn.DataParallel(self.model.cuda(),
                                        device_ids=self.opt.gpu_ids,
                                        output_device=self.opt.gpu_ids[-1],
                                        )
            self.models = [self.model]
            self.names = [self.opt.name]
            self.save_path = create_path(create_path(self.opt.save_path, self.opt.name), self.opt.dataset)
            self.save_paths = [self.save_path]
        else:
            self.models = []
            self.names = []
            self.save_paths = []
            for name in self.opt.names.split(','):
                self.opt.name = name
                self.model = TestModel(self.opt)
                self.model.eval()
                self.model = torch.nn.DataParallel(self.model.cuda(),
                                            device_ids=self.opt.gpu_ids,
                                            output_device=self.opt.gpu_ids[-1],
                                            )
                self.models.append(self.model)
                self.names.append(name)
                self.save_path = create_path(create_path(self.opt.save_path, self.opt.name), self.opt.dataset)
                self.save_paths.append(self.save_path)

        # test
        self.dataloader_iterator = iter(self.dataloaders[0])

        self.fitting_model, self.alignment_model = load_3ddfa(self.opt)


    def synthesize_image(self, img_fp_suffix):
        img_fp = os.path.join(self.opt.img_prefix, img_fp_suffix)
        param, landmarks, img_orig, yaw_pose = get_param(
            self.fitting_model, self.alignment_model, img_fp, self.opt
        )
        if img_orig is None:
            raise Exception('No Image')
        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        wrapped_img, M = affine_align(img, landmarks)
        M = torch.from_numpy(M).float()
        data = get_input(wrapped_img, M, img_fp, self.render_layer_list[0], rotate_yaw_pose(yaw_pose), param)

        img_path = data['path']
        poses = data['pose_list']


        generate_rotateds = []
        for model in self.models:
            generate_rotated = model.forward(data, mode='single')
            generate_rotateds.append(generate_rotated)

        for n, name in enumerate(self.names):
            self.opt.name = name
            for b in range(generate_rotateds[n].shape[0]):
                # get savepaths
                rotated_file_savepath = create_paths(self.save_paths[n], img_path[b], folderlevel=self.folderlevel, pose=poses[b])
                save_img(generate_rotateds[n][b], rotated_file_savepath)




