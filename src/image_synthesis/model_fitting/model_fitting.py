#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import math
from tqdm import tqdm
import time
import face_alignment
from .utils.ddfa import ToTensorGjz, NormalizeGjz
import scipy.io as sio
from .utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors, get_aligned_param, get_5lmk_from_68lmk
from .utils.estimate_pose import parse_pose
from .utils.params import param_mean, param_std
from .utils.render import get_depths_image, cget_depths_image, cpncc, crender_colors
from .utils.paf import gen_img_paf
import model_fitting.mobilenet_v1 as mobilenet_v1
import torch.backends.cudnn as cudnn

STD_SIZE = 120


def load_3ddfa(args):
    # 1. load pre-trained model
    checkpoint_fp = 'model_fitting/models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    with open(args.img_list) as f:
        img_list = [x.strip() for x in f.readlines()]

    alignment_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    return model, alignment_model, img_list

def get_param(model, alignment_model, img_fp, args):

      img_ori = cv2.imread(os.path.join(args.img_prefix, img_fp))

      transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

      pts_res = []
      Ps = []  # Camera matrix collection
      poses = []  # pose collection, [todo: validate it]
      vertices_lst = []  # store multiple face vertices
      ind = 0
      suffix = get_suffix(img_fp)

      # face alignment model use RGB as input, result is a tuple with landmarks and boxes
      preds = alignment_model.get_landmarks(img_ori[:, :, ::-1])
      pts_2d_68 = preds[0]
      #print(preds[0])
      pts_2d_5 = get_5lmk_from_68lmk(pts_2d_68)
      roi_box = parse_roi_box_from_landmark(pts_2d_68.T)

      img = crop_img(img_ori, roi_box)
      # import pdb; pdb.set_trace()

      # forward: one step
      img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
      input = transform(img).unsqueeze(0)
      with torch.no_grad():
          if args.mode == 'gpu':
              input = input.cuda()
          param = model(input)
          param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

      pts68 = predict_68pts(param, roi_box)

      # two-step for more accurate bbox to crop face
      if args.bbox_init == 'two':
          roi_box = parse_roi_box_from_landmark(pts68)
          img_step2 = crop_img(img_ori, roi_box)
          img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
          input = transform(img_step2).unsqueeze(0)
          with torch.no_grad():
              if args.mode == 'gpu':
                  input = input.cuda()
              param = model(input)
              param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

      pose = parse_pose(param)[1][0]
      this_param = param * param_std + param_mean
      this_param = np.concatenate((this_param, roi_box))
      
      
      return this_param, pose

