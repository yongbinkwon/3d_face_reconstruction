import sys
import argparse
import math
import os
from ..util import util
import torch
from ..models import get_option_setter as get_modelOption_setter
import pickle
from ..model_fitting.utils.ddfa import str2bool


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--name', type=str, default='rotate_synthesis', help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--gpu_ids', type=str, default='0', help='useless')
        parser.add_argument('--checkpoints_dir', type=str, default='image_synthesis/checkpoints', help='models are saved here')
        parser.add_argument('--model', type=str, default='rotatespade', help='which model to use, rotate|rotatespade')
        parser.add_argument('--trainer', type=str, default='rotate', help='which trainer to use, rotate|rotatespade')
        parser.add_argument('--norm_G', type=str, default='spectralsyncbatch', help='instance normalization or batch normalization')
        parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--device_count', type=int, default=1, help='the total number of gpus to use')
        parser.add_argument('--render_thread', type=int, default=1, help='number of gpus used for rendering')
        parser.add_argument('--chunk_size', default=[1], type=int, nargs='+', help='specify the batch size on each training gpu. Training gpu # = device_count - render_thread')
        # input/output sizes
        # parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--preprocess_mode', type=str, default='resize_and_crop', help='scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
        parser.add_argument('--load_size', type=int, default=400, help='Scale images to this size. The final image will be cropped to --crop_size.')
        parser.add_argument('--crop_size', type=int, default=256, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--label_nc', type=int, default=5, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
        parser.add_argument('--contain_dontcare_label', action='store_true', help='if the label map contains dontcare label (dontcare=255)')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        parser.add_argument('--landmark_align', action='store_true', help='wether there is landmark_align')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--nThreads', default=3, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
        parser.add_argument('--cache_filelist_write', action='store_true', help='saves the current filelist into a text file, so that it loads faster')
        parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')

        # for displays
        parser.add_argument('--display_winsize', type=int, default=400, help='display window size')

        # for generator
        parser.add_argument('--netG', type=str, default='rotatespade', help='selects model to use for netG (pix2pixhd | spade)')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

        # for instance-wise features
        parser.add_argument('--no_instance', action='store_true', default='True', help='if specified, do *not* add instance map as input')
        parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')

        parser.add_argument('--no_gaussian_landmark', action='store_false', help='whether to use no_gaussian_landmark (1.0 landmark) for rotatespade model')
        parser.add_argument('--label_mask', action='store_false', help='whether to use face mask')
        parser.add_argument('--heatmap_size', type=float, default=2.5, help='the size of the heatmap, used in rotatespade model')
        parser.add_argument('--erode_kernel', type=int, default=21, help='erode kernel size, used in renderer')

        #3ddfa
        parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
        parser.add_argument('--bbox_init', default='two', type=str,
                            help='one|two: one-step bbox initialization or two-step')
        parser.add_argument('--dump_2d_img', default='true', type=str2bool, help='whether to save 3d rendered image')
        parser.add_argument('--dump_param', default='true', type=str2bool, help='whether to save param')
        parser.add_argument('--dump_lmk', default='true', type=str2bool, help='whether to save landmarks')
        parser.add_argument('--save_dir', default='results', type=str, help='dir to save result')
        parser.add_argument('--save_lmk_dir', default='image_synthesis/model_fitting/example', type=str, help='dir to save landmark result')
        parser.add_argument('--img_list', default='image_synthesis/model_fitting/example/file_list.txt', type=str, help='test image list file')
        parser.add_argument('--img_prefix', default='image_synthesis/model_fitting/example/Images', type=str, help='test image prefix')
        parser.add_argument('--rank', default=0, type=int, help='used when parallel run')
        parser.add_argument('--world_size', default=1, type=int, help='used when parallel run')
        parser.add_argument('--resume_idx', default=0, type=int)

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = get_modelOption_setter(model_name)
        parser = model_option_setter(parser)

        """
        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        dataset_option_setter = get_dataOption_setter(dataset_mode)
        parser = dataset_option_setter(parser)
        """

        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # lt options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()

        self.print_options(opt)

        if opt.yaw_poses is not None:
            for pose in opt.yaw_poses:
                assert abs(pose) <= 90, "yaw pose must be between [-90, 90]"
            opt.yaw_poses = [round(x / 180.0 * math.pi, 2) for x in opt.yaw_poses]
        if opt.pitch_poses is not None:
            for pose in opt.pitch_poses:
                assert abs(pose) <= 90, "pitch pose must be between [-90, 90]"
            opt.pitch_poses = [round(x / 180.0 * math.pi, 2) for x in opt.pitch_poses]

        # Set semantic_nc based on the option.
        # This will be convenient in many places
        opt.semantic_nc = opt.label_nc 

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
        #     "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
        #     % (opt.batchSize, len(opt.gpu_ids))
        opt.batchSize = sum(opt.chunk_size)


        self.opt = opt
        return self.opt
