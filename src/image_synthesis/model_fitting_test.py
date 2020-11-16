from options.test_options import TestOptions
from model_fitting.model_fitting import load_3ddfa, get_param
from tqdm import tqdm

if __name__ == '__main__':
    opt = TestOptions().parse()
    fitting_model, alignment_model, img_list = load_3ddfa(opt)
    for img_idx, img_fp in enumerate(tqdm(img_list)):
        param, yaw_pose = get_param(fitting_model, alignment_model, img_fp, opt)
