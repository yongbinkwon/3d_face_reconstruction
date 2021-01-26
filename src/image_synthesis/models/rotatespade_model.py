import torch
from . import networks
from ..util import util
from ..data import curve
import numpy as np
import os


class RotateSPADEModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser):
        networks.modify_commandline_options(parser)
        return parser

    def __init__(self, opt):
        super(RotateSPADEModel, self).__init__()
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        self.real_image = torch.zeros(opt.batchSize, 3, opt.crop_size, opt.crop_size)
        self.input_semantics = torch.zeros(opt.batchSize, 3, opt.crop_size, opt.crop_size)

        self.netG, self.netD, self.netD_rotate = self.initialize_networks(opt)


    def landmark_68_to_5(self, t68):
        le = t68[36:42, :].mean(axis=0, keepdims=True)
        re = t68[42:48, :].mean(axis=0, keepdims=True)
        no = t68[31:32, :]
        lm = t68[48:49, :]
        rm = t68[54:55, :]
        t5 = np.concatenate([le, re, no, lm, rm], axis=0)
        t5 = t5.reshape(10)
        t5 = torch.from_numpy(t5).unsqueeze(0).cuda()
        return t5

    def get_seg_map(self, landmarks, no_guassian=False, size=256, original_angles=None):
        landmarks = landmarks[:, :, :2].cpu().numpy().astype(np.float)
        all_heatmap = []
        all_orig_heatmap = []
        if original_angles is None:
            original_angles = torch.zeros(landmarks.shape[0])
        # key_points = []
        for i in range(landmarks.shape[0]):
            heatmap = curve.points_to_heatmap_68points(landmarks[i], 13, size, self.opt.heatmap_size)

            heatmap2 = curve.combine_map(heatmap, no_guassian=no_guassian)
            if torch.abs(original_angles[i]) < 0.255:
                heatmap = np.zeros_like(heatmap)

            all_heatmap.append(heatmap2)
            all_orig_heatmap.append(heatmap)
            # key_points.append(self.landmark_68_to_5(landmarks[i]))
        all_heatmap = np.stack(all_heatmap, axis=0)
        all_orig_heatmap = np.stack(all_orig_heatmap, axis=0)
        all_heatmap = torch.from_numpy(all_heatmap.astype(np.float32)).cuda()
        all_orig_heatmap = torch.from_numpy(all_orig_heatmap.astype(np.float32)).cuda()
        all_orig_heatmap = all_orig_heatmap.permute(0, 3, 1, 2)
        all_orig_heatmap[all_orig_heatmap > 0] = 2.0
        return all_heatmap, all_orig_heatmap

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    # |data|: dictionary of the input data


    def forward(self, data, mode):
        real_image = data['image']

        orig_landmarks = data['orig_landmarks']
        rotated_landmarks = data['rotated_landmarks']
        original_angles = data['original_angles']
        self.orig_seg, orig_seg_all = \
            self.get_seg_map(orig_landmarks, self.opt.no_gaussian_landmark, self.opt.crop_size, original_angles)
        self.rotated_seg, rotated_seg_all = \
            self.get_seg_map(rotated_landmarks, self.opt.no_gaussian_landmark, self.opt.crop_size, original_angles)

        input_semantics = data['mesh']
        rotated_mesh = data['rotated_mesh']
        if self.opt.label_mask:

            input_semantics = (input_semantics + orig_seg_all[:, 4].unsqueeze(1) + orig_seg_all[:, 0].unsqueeze(1))
            rotated_mesh = (rotated_mesh + rotated_seg_all[:, 4].unsqueeze(1) + rotated_seg_all[:, 0].unsqueeze(1))
            input_semantics[input_semantics >= 1] = 0
            rotated_mesh[rotated_mesh >= 1] = 0

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, self.orig_seg, netD=self.netD, mode=mode, no_ganFeat_loss=self.opt.no_ganFeat_loss,
                    no_vgg_loss=self.opt.no_vgg_loss, lambda_D=self.opt.lambda_D)
            return g_loss, generated, input_semantics
        if mode == 'generator_rotated':
            g_loss, generated = self.compute_generator_loss(
                rotated_mesh, real_image, self.rotated_seg, netD=self.netD_rotate, mode=mode, no_ganFeat_loss=True,
                    no_vgg_loss=self.opt.no_vgg_loss, lambda_D=self.opt.lambda_rotate_D)
            return g_loss, generated, rotated_mesh
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, self.orig_seg, netD=self.netD, lambda_D=self.opt.lambda_D)
            return d_loss
        elif mode == 'discriminator_rotated':
            d_loss = self.compute_discriminator_loss(
                rotated_mesh, real_image, self.rotated_seg, self.netD_rotate, lambda_D=self.opt.lambda_rotate_D)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                if self.opt.label_mask:
                    rotated_mesh = (
                                rotated_mesh + rotated_seg_all[:, 4].unsqueeze(1) + rotated_seg_all[:, 0].unsqueeze(1))
                    rotated_mesh[rotated_mesh >= 1] = 0
                fake_image = self.generate_fake(input_semantics, real_image, self.orig_seg)
                fake_rotate = self.generate_fake(rotated_mesh, real_image, self.rotated_seg)

            return fake_image, fake_rotate
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.train_rotate:
            util.save_network(self.netD_rotate, 'D_rotate', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################


    def initialize_networks(self, opt):

        netG = networks.define_G(opt)
        netD = None
        netD_rotate = None
        pretrained_path = ''
        
        self.load_network(netG, 'G', opt.which_epoch, pretrained_path)

        return netG, netD, netD_rotate

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding

    def compute_generator_loss(self, input_semantics, real_image, seg, netD, mode, no_ganFeat_loss=False, no_vgg_loss=False, lambda_D=1):
        G_losses = {}

        fake_image = self.generate_fake(
            input_semantics, real_image, seg)

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image, seg, netD)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False) * lambda_D

        if not no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    if j == 0:
                        unweighted_loss *= self.opt.lambda_image
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not no_vgg_loss:
            if mode == 'generator_rotated':
                num = 2
            else:
                num = 0
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image, num) \
                    * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image, seg, netD, lambda_D=1):
        D_losses = {}
        with torch.no_grad():
            fake_image = self.generate_fake(input_semantics, real_image, seg)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real= self.discriminate(
            input_semantics, fake_image, real_image, seg, netD)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True) * lambda_D

        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True) * lambda_D

        return D_losses

    def generate_fake(self, input_semantics, real_image, seg):

        fake_image = self.netG(input_semantics, seg)
        
        return fake_image

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image, seg, netD):
        if self.opt.D_input == "concat":
            fake_concat = torch.cat([seg, fake_image], dim=1)
            real_concat = torch.cat([self.orig_seg, real_image], dim=1)
        else:
            fake_concat = fake_image
            real_concat = real_image

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            # rotate_fake = pred[pred.size(0) // 3: pred.size(0) * 2 // 3]
            real = pred[pred.size(0)//2 :]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def load_separately(self, network, network_label, opt):
        load_path = None
        if network_label == 'G':
            load_path = opt.G_pretrain_path
        elif network_label == 'D':

            load_path = opt.D_pretrain_path
        elif network_label == 'D_rotate':
            load_path = opt.D_rotate_pretrain_path
        elif network_label == 'E':
            load_path = opt.E_pretrain_path

        if load_path is not None:
            if os.path.isfile(load_path):
                print("=> loading checkpoint '{}'".format(load_path))
                checkpoint = torch.load(load_path)
                util.copy_state_dict(checkpoint, network)
        else:
            print("no load_path")
        return network

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        print("save_path")
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise ('Generator must exist!')
        else:
            # network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print(
                            'Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    not_initialized = set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
