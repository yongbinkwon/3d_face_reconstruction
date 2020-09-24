import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from models.networks.sync_batchnorm import DataParallelWithCallback
import sys
import numpy as np
import os
import data
from util.iter_counter import IterationCounter
from options.test_options import TestOptions
from models.test_model import TestModel
from util.visualizer import Visualizer
from util import html, util
from torch.multiprocessing import Process, Queue, Pool
from data.data_utils import init_parallel_jobs
from skimage import transform as trans
import cv2
import time
import torch
from models.networks.rotate_render import TestRender
from tqdm import tqdm
from data.data_utils import get_multipose_test_input, get_test_input
from model_fitting.model_fitting import load_3ddfa, get_param



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


if __name__ == '__main__':


    opt = TestOptions().parse()

    data_info = data.dataset_info()
    datanum = data_info.get_dataset(opt)[0]
    folderlevel = data_info.folder_level[datanum]

    dataloaders = data.create_dataloader_test(opt)

    ngpus = opt.device_count

    render_gpu_ids = list(range(ngpus - opt.render_thread, ngpus))
    render_layer_list = []
    for gpu in render_gpu_ids:
        opt.gpu_ids = gpu
        render_layer = TestRender(opt)
        render_layer_list.append(render_layer)

    opt.gpu_ids =  list (range (0, ngpus))
    print('Testing gpu ', opt.gpu_ids)
    if opt.names is None:
        model = TestModel(opt)
        model.eval()
        model = torch.nn.DataParallel(model.cuda(),
                                      device_ids=opt.gpu_ids,
                                      output_device=opt.gpu_ids[-1],
                                      )
        models = [model]
        names = [opt.name]
        save_path = create_path(create_path(opt.save_path, opt.name), opt.dataset)
        save_paths = [save_path]
    else:
        models = []
        names = []
        save_paths = []
        for name in opt.names.split(','):
            opt.name = name
            model = TestModel(opt)
            model.eval()
            model = torch.nn.DataParallel(model.cuda(),
                                          device_ids=opt.gpu_ids,
                                          output_device=opt.gpu_ids[-1],
                                          )
            models.append(model)
            names.append(name)
            save_path = create_path(create_path(opt.save_path, opt.name), opt.dataset)
            save_paths.append(save_path)

    # test
    dataloader_iterator = iter(dataloaders[0])

    fitting_model, alignment_model, img_list = load_3ddfa(opt)
    for img_idx, img_fp in enumerate(tqdm(img_list)):

      param, yaw_pose = get_param(fitting_model, alignment_model, img_fp, opt)
      data = get_rotated_test_input(next(dataloader_iterator), render_layer_list[0], rotate_yaw_pose(yaw_pose), param)

      img_path = data['path']
      poses = data['pose_list']


      generate_rotateds = []
      for model in models:
          generate_rotated = model.forward(data, mode='single')
          generate_rotateds.append(generate_rotated)

      for n, name in enumerate(names):
          opt.name = name
          for b in range(generate_rotateds[n].shape[0]):
              # get savepaths
              rotated_file_savepath = create_paths(save_paths[n], img_path[b], folderlevel=folderlevel, pose=poses[b])
              image_numpy = save_img(generate_rotateds[n][b], rotated_file_savepath)


"""
    except KeyboardInterrupt:
        print("Interrupted!")
        for fs in f:
            fs.close()
        pass

    except Exception as e:
        print(e)
        for fs in f:
            fs.close()

    else:
        print('finished')
        for fs in f:
            fs.close()
"""




