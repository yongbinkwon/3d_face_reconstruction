import numpy as np
import os
import data
from options.test_options import TestOptions
from models.test_model import TestModel
from util import util
import torch
from models.networks.rotate_render import TestRender
from tqdm import tqdm
from data.data_utils import get_input
from model_fitting.model_fitting import load_3ddfa, get_param



class Synthesize():

    def __init__(self):
        
        self.opt = TestOptions().parse()

        self.data_info = data.dataset_info()
        self.datanum = self.data_info.get_dataset(opt)[0]
        self.folderlevel = self.data_info.folder_level[datanum]

        self.dataloaders = data.create_dataloader_test(opt)

        self.ngpus = opt.device_count

        self.render_gpu_ids = list(range(ngpus - opt.render_thread, ngpus))
        self.render_layer_list = []
        for gpu in self.render_gpu_ids:
            self.opt.gpu_ids = gpu
            render_layer = TestRender(opt)
            render_layer_list.append(render_layer)

        self.opt.gpu_ids =  list (range (0, ngpus))
        print('Testing gpu ', opt.gpu_ids)
        if opt.names is None:
            self.model = TestModel(opt)
            self.model.eval()
            self.model = torch.nn.DataParallel(model.cuda(),
                                        device_ids=opt.gpu_ids,
                                        output_device=opt.gpu_ids[-1],
                                        )
            self.models = [model]
            self.names = [opt.name]
            self.save_path = create_path(create_path(opt.save_path, opt.name), opt.dataset)
            self.save_paths = [save_path]
        else:
            self.models = []
            self.names = []
            self.save_paths = []
            for name in opt.names.split(','):
                self.opt.name = name
                self.model = TestModel(opt)
                self.model.eval()
                self.model = torch.nn.DataParallel(model.cuda(),
                                            device_ids=opt.gpu_ids,
                                            output_device=opt.gpu_ids[-1],
                                            )
                self.models.append(model)
                self.names.append(name)
                self.save_path = create_path(create_path(opt.save_path, opt.name), opt.dataset)
                self.save_paths.append(save_path)

        # test
        self.dataloader_iterator = iter(dataloaders[0])

        self.fitting_model, self.alignment_model, self.img_list = load_3ddfa(opt)

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


    def synthesize_image(self, img_fp):

        param, landmarks, img_ori, yaw_pose = get_param(self.fitting_model, self.alignment_model, img_fp, self.opt)
        data = get_input(next(dataloader_iterator), render_layer_list[0], rotate_yaw_pose(yaw_pose), param)

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
                rotated_file_savepath = create_paths(save_paths[n], img_path[b], folderlevel=folderlevel, pose=poses[b])
                image_numpy = save_img(generate_rotateds[n][b], rotated_file_savepath)




