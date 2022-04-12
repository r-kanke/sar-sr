import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

from . import rcan
from . import scan
from . import discriminator
from . import srgan

class FeatureExtractor(nn.Module):
    """CNN to extract feature for perceputual loss.

    Vgg19 is used as feature extractor.
    
    Notes:
        Vgg19 expects color (3 channel) input. When applying grayscale,
        color channel is repeated to adjust 3.
    """
    def __init__(self, weight_path=None, conv_index=54):
        super().__init__()

        if weight_path is None:
            vgg19_model = torchvision.models.vgg19(pretrained=True)
        else:
            vgg19_model = torchvision.models.vgg19(pretrained=False)
            vgg19_model.load_state_dict(torch.load(weight_path))

        if conv_index == 22:
            self.vgg19 = nn.Sequential(*list(
                vgg19_model.features.children())[:8])

        elif conv_index == 54:
            self.vgg19 = nn.Sequential(*list(
                vgg19_model.features.children())[:35])

    def forward(self, im):
        # grayscale to rgb
        if im.ndim == 3 and len(im[0]) == 1:
            im = im.repeat(3, 1, 1)  # (1, H, W) -> (3, H, W)
        elif im.ndim == 4 and len(im[1]) == 1:
            im = im.repeat(1, 3, 1, 1)  # (B, 1, H, W) -> (B, 3, H, W)

        return self.vgg19(im)


class GAN(nn.Module):
    def __init__(self, opt, cal_psnr_ssim, unnorm):
        super().__init__()

        self.loss_ratio = opt.loss_ratio
        self.parallel_processing = opt.parallel_processing
        self.cal_psnr_ssim = cal_psnr_ssim
        self.unnorm = unnorm
        self.s1_data_range = opt.s1_data_range
        self.use_optical = (opt.n_colors == 4)

        # generator
        if opt.model_name == 'rcan':
            self.generator = rcan.RCAN(
                scale=opt.scale, n_resblocks=opt.n_resblocks, n_feats=opt.n_feats, n_resgroups=opt.n_resgroups,
                n_colors=opt.n_colors, n_out_colors=opt.n_out_colors, act=nn.LeakyReLU(0.2, inplace=True),
                ).to(opt.device)

        elif opt.model_name == 'scan':
            self.generator = scan.SCAN(
                scale=opt.scale, base_n_resblocks=opt.base_n_resblocks, base_n_feats=opt.base_n_feats, base_n_resgroups=opt.base_n_resgroups,
                branch_n_feats=opt.branch_n_feats, branch_n_resblocks=opt.branch_n_resblocks, branch_n_feats_sa_mid=opt.branch_n_feats_sa_mid,
                n_colors=opt.n_colors, n_out_colors=opt.n_out_colors, act=nn.LeakyReLU(0.2, inplace=True),
                ).to(opt.device)
        
        elif opt.model_name == 'srgan':
            self.generator = srgan.Generator(
                in_channels=opt.n_colors, out_channels=opt.n_out_colors
                ).to(opt.device)

        # discriminator
        if opt.model_name == 'rcan' or opt.model_name == 'scan':
            self.discriminator = discriminator.Discriminator(
                input_shape=opt.out_shape, edge_extractor_opt=opt.edge_extractor_opt, patch_gan=True,
                ).to(opt.device)
        elif opt.model_name == 'srgan':
            self.discriminator = discriminator.Discriminator(
                input_shape=opt.out_shape, edge_extractor_opt=opt.edge_extractor_opt, patch_gan=False,
                ).to(opt.device)

        # parallel processing
        if self.parallel_processing:
            self.generator = torch.nn.DataParallel(self.generator)
            self.discriminator = torch.nn.DataParallel(self.discriminator)

        # loss function
        self.criterion_pixel = nn.MSELoss().to(opt.device)  # when l1 loss: nn.L1Loss().to(opt.device)
        self.criterion_adv = nn.BCEWithLogitsLoss().to(opt.device)

        # feature extractor
        if self.loss_ratio['content'] != 0.:
            # srgan-defined content loss instead of mine:
            self.criterion_content = srgan.ContentLoss(opt.feature_extractor_weight_path).to(opt.device)
            # mine:
            # self.feature_extractor = FeatureExtractor(weight_path=opt.feature_extractor_weight_path, conv_index=54).to(opt.device)
            # self.feature_extractor.eval()
            # if self.parallel_processing:
            #     self.feature_extractor = torch.nn.DataParallel(self.feature_extractor)
            # self.criterion_content = nn.L1Loss().to(opt.device)

        # optimizer
        params_g = self.generator.module.parameters() if self.parallel_processing else self.generator.parameters()
        params_d = self.discriminator.module.parameters() if self.parallel_processing else self.generator.parameters()
        self.optim_g = torch.optim.Adam(params=params_g, lr=opt.lr_g, betas=(opt.b1, opt.b2))
        self.optim_d = torch.optim.SGD(params=params_d, lr=opt.lr_d)

        # scheduler
        self.scheduler_g = torch.optim.lr_scheduler.MultiStepLR(self.optim_g, milestones=[], gamma=opt.lr_gamma)
        self.scheduler_d = torch.optim.lr_scheduler.MultiStepLR(self.optim_d, milestones=[], gamma=opt.lr_gamma)

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.writer = SummaryWriter(log_dir=opt.tb_log_dir)

    def train_g(self, batch):
        """Train generator only using pixel loss.

        Train using one mini-batch images.

        Args:
            batch (Tensor): [description]
        """
        # preproces
        if self.use_optical:
            ims_lr = torch.autograd.Variable(torch.cat((batch.lows1, batch.lows2), 1).type(dtype=self.Tensor, non_blocking=True))
        else:
            ims_lr = torch.autograd.Variable(batch.lows1.type(dtype=self.Tensor, non_blocking=True))
        ims_hr = torch.autograd.Variable(batch.highs1.type(dtype=self.Tensor, non_blocking=True))

        # set train mode
        self.generator.train()

        # forward prop
        gen_hr = self.generator(ims_lr)

        # calc loss
        loss_pixel = self.criterion_pixel(gen_hr, ims_hr)

        if self.loss_ratio['content'] != 0.:
            # srgan:
            unnorm_gen_hr = self.unnorm(gen_hr.detach().clone())
            unnorm_ims_hr = self.unnorm(ims_hr.detach().clone())
            loss_content = self.criterion_content(unnorm_gen_hr, unnorm_ims_hr)
            # mine:
            # gen_feature = self.feature_extractor(gen_hr)
            # real_feature = self.feature_extractor(ims_hr).detach()
            # loss_content = self.criterion_content(gen_feature, real_feature)
        else:
            loss_content = self.Tensor(np.zeros(1))

        loss_g = self.loss_ratio['pixel'] * loss_pixel + self.loss_ratio['content'] * loss_content

        # back prop
        self.optim_g.zero_grad()
        loss_g.backward()
        self.optim_g.step()
        self.scheduler_g.step()

        # calc psnr and ssim
        psnr, ssim = self.cal_psnr_ssim(gen_hr, ims_hr, data_range=self.s1_data_range)

        return {'train_loss_g': loss_g.item(),'train_loss_pixel': loss_pixel.item(), 'train_loss_content': loss_content.item(),
                'train_psnr': psnr, 'train_ssim': ssim, 'lr_g': self.scheduler_g.get_last_lr()[0],}

    def train_d(self, batch):
        """Train discriminator using one mini-batch images.

        Args:
            batch (Tensor): [description]
        """
        if self.use_optical:
            ims_lr = torch.autograd.Variable(torch.cat((batch.lows1, batch.lows2), 1).type(dtype=self.Tensor, non_blocking=True))
        else:
            ims_lr = torch.autograd.Variable(batch.lows1.type(dtype=self.Tensor, non_blocking=True))
        ims_hr = torch.autograd.Variable(batch.highs1.type(dtype=self.Tensor, non_blocking=True))

        # ground truth
        output_shape = self.discriminator.module.output_shape if self.parallel_processing else self.discriminator.output_shape

        ones = np.random.rand(ims_lr.size(0), *output_shape) * 0.5 + 0.7  # random [0.7, 1.2)
        zeros = np.random.rand(ims_lr.size(0), *output_shape) * 0.3  # random [0.0, 0.3)

        valid = torch.autograd.Variable(self.Tensor(ones), requires_grad=False)
        fake = torch.autograd.Variable(self.Tensor(zeros), requires_grad=False)

        # set train mode
        self.generator.eval()
        self.discriminator.train()

        # forward prop
        gen_hr = self.generator(ims_lr)

        # discriminator loss
        pred_real = self.discriminator(ims_hr)
        pred_fake = self.discriminator(gen_hr.detach())

        loss_real = self.criterion_adv(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = self.criterion_adv(pred_fake - pred_real.mean(0, keepdim=True), fake)
        loss_d = (loss_real + loss_fake) / 2  # devide by 2 to slow-down D's learning

        # backporp
        self.optim_d.zero_grad()
        loss_d.backward()
        self.optim_d.step()
        self.scheduler_d.step()

        # save result
        train_info = {'train_loss_d': loss_d.item(), 'lr_d': self.scheduler_d.get_last_lr()[0],}

        return train_info

    def train(self, batch):
        """Train using all losses.

        Args:
            batch (Tensor): [description]
        """
        if self.use_optical:
            ims_lr = torch.cat((batch.lows1, batch.lows2), 1)
        else:
            ims_lr = batch.lows1
        ims_lr = torch.autograd.Variable(ims_lr.type(dtype=self.Tensor, non_blocking=True))
        ims_hr = torch.autograd.Variable(batch.highs1.type(dtype=self.Tensor, non_blocking=True))

        # ground truth
        output_shape = self.discriminator.module.output_shape if self.parallel_processing else self.discriminator.output_shape

        ones = np.random.rand(ims_lr.size(0), *output_shape) * 0.5 + 0.7  # random [0.7, 1.2)
        zeros = np.random.rand(ims_lr.size(0), *output_shape) * 0.3  # random [0.0, 0.3)

        valid = torch.autograd.Variable(self.Tensor(ones), requires_grad=False)
        fake = torch.autograd.Variable(self.Tensor(zeros), requires_grad=False)

        # set train mode
        self.generator.train()
        self.discriminator.train()

        # forward prop
        gen_hr = self.generator(ims_lr)

        # prediction: discriminator
        pred_real = self.discriminator(ims_hr).detach()
        pred_fake = self.discriminator(gen_hr)

        # pixel loss
        loss_pixel = self.criterion_pixel(gen_hr, ims_hr)

        # adversarial loss
        loss_adv = self.criterion_adv(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # perceputual loss
        if self.loss_ratio['content'] != 0.:
            # srgan:
            unnorm_gen_hr = self.unnorm(gen_hr.detach().clone())
            unnorm_ims_hr = self.unnorm(ims_hr.detach().clone())
            loss_content = self.criterion_content(unnorm_gen_hr, unnorm_ims_hr)
            # mine:
            # gen_feature = self.feature_extractor(gen_hr)
            # real_feature = self.feature_extractor(ims_hr).detach()
            # loss_content = self.criterion_content(gen_feature, real_feature)
        else:
            loss_content = self.Tensor(np.zeros(1))

        # generator loss
        loss_g = self.loss_ratio['pixel'] * loss_pixel + self.loss_ratio['content'] * loss_content + self.loss_ratio['adv'] * loss_adv

        # discriminator loss
        pred_real = self.discriminator(ims_hr)
        pred_fake = self.discriminator(gen_hr.detach())

        loss_real = self.criterion_adv(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = self.criterion_adv(pred_fake - pred_real.mean(0, keepdim=True), fake)
        loss_d = (loss_real + loss_fake) / 2  # devide by 2 to slow-down D's learning

        # backporp
        self.optim_g.zero_grad()
        self.optim_d.zero_grad()
        loss_g.backward()
        loss_d.backward()
        self.optim_g.step()
        self.optim_d.step()
        self.scheduler_g.step()
        self.scheduler_d.step()

        # calc psnr and ssim
        psnr, ssim = self.cal_psnr_ssim(gen_hr, ims_hr, data_range=self.s1_data_range)

        # save result
        train_info = {'train_loss_d': loss_d.item(),
                      'train_loss_g': loss_g.item(),
                      'train_loss_pixel': loss_pixel.item(),
                      'train_loss_content': loss_content.item(),
                      'train_loss_adv': loss_adv.item(),
                      'train_psnr': psnr,
                      'train_ssim': ssim,
                      'lr_g': self.scheduler_g.get_last_lr()[0],
                      'lr_d': self.scheduler_d.get_last_lr()[0],
                      }

        return train_info

    def validate(self, loader):
        """Validate using all losses.

        Args:
            loader (torch.utils.data.DataLoader): [description]
        """
        val_info = {'val_loss_d': 0., 'val_loss_g': 0., 'val_loss_content': 0., 'val_loss_adv': 0., 'val_loss_pixel': 0.,
                    'val_psnr': 0., 'val_ssim': 0.,}

        # set evaluation mode
        self.generator.eval()
        self.discriminator.eval()

        with torch.no_grad():
            for batch in loader:
                if self.use_optical:
                    ims_lr = torch.autograd.Variable(torch.cat((batch.lows1, batch.lows2), 1).type(dtype=self.Tensor, non_blocking=True))
                else:
                    ims_lr = torch.autograd.Variable(batch.lows1.type(dtype=self.Tensor, non_blocking=True))
                ims_hr = torch.autograd.Variable(batch.highs1.type(dtype=self.Tensor, non_blocking=True))

                # ground truth
                output_shape = self.discriminator.module.output_shape if self.parallel_processing else self.discriminator.output_shape
                valid = torch.autograd.Variable(self.Tensor(np.ones((ims_lr.size(0), *output_shape))), requires_grad=False)
                fake = torch.autograd.Variable(self.Tensor(np.zeros((ims_lr.size(0), *output_shape))), requires_grad=False)

                # forward prop
                gen_hr = self.generator(ims_lr)

                # prediction: discriminator
                pred_real = self.discriminator(ims_hr).detach()
                pred_fake = self.discriminator(gen_hr)

                # pixel loss
                loss_pixel = self.criterion_pixel(gen_hr, ims_hr)

                # perceputual loss
                if self.loss_ratio['content'] != 0.:
                    # srgan:
                    unnorm_gen_hr = self.unnorm(gen_hr.detach().clone())
                    unnorm_ims_hr = self.unnorm(ims_hr.detach().clone())
                    loss_content = self.criterion_content(unnorm_gen_hr, unnorm_ims_hr)
                    # mine:
                    # gen_feature = self.feature_extractor(gen_hr)
                    # real_feature = self.feature_extractor(ims_hr).detach()
                    # loss_content = self.criterion_content(gen_feature, real_feature)
                else:
                    loss_content = self.Tensor(np.zeros(1))

                # adversarial loss
                loss_adv = self.criterion_adv(pred_fake - pred_real.mean(0, keepdim=True), valid)

                # generator loss
                loss_g = self.loss_ratio['pixel'] * loss_pixel + self.loss_ratio['adv'] * loss_adv + self.loss_ratio['content'] * loss_content

                # discriminator loss
                pred_real = self.discriminator(ims_hr)
                pred_fake = self.discriminator(gen_hr.detach())

                loss_real = self.criterion_adv(pred_real - pred_fake.mean(0, keepdim=True), valid)
                loss_fake = self.criterion_adv(pred_fake - pred_real.mean(0, keepdim=True), fake)
                loss_d = (loss_real + loss_fake) / 2  # devide by 2 to slow-down D's learning

                # calc psnr and ssim
                psnr, ssim = self.cal_psnr_ssim(gen_hr, ims_hr, data_range=self.s1_data_range)

                # save result
                val_info['val_loss_d'] += loss_d / len(loader)
                val_info['val_loss_g'] += loss_g / len(loader)
                val_info['val_loss_content'] += loss_content / len(loader)
                val_info['val_loss_adv'] += loss_adv / len(loader)
                val_info['val_loss_pixel'] += loss_pixel / len(loader)
                val_info['val_psnr'] += psnr / len(loader)
                val_info['val_ssim'] += ssim / len(loader)

        return val_info

    def save_loss(self, info, batches_done):
        """Save loss.

        Args:
            train_info ([type]): [description]
            batches_done ([type]): [description]
        """
        for k, v in info.items():
            self.writer.add_scalar(k, v, batches_done)

    def save_image(self, batch, batches_done):
        raise NotImplementedError

    def save_weight(self, batches_done, opt):
        """Save weight.

        Args:
            batches_done ([type]): [description]
        """
        generator_weight_path = os.path.splitext(opt.model_path)[0] + '_{:08}_g'.format(batches_done) + os.path.splitext(opt.model_path)[1]
        discriminator_weight_path = os.path.splitext(opt.model_path)[0] + '_{:08}_d'.format(batches_done) + os.path.splitext(opt.model_path)[1]

        if self.parallel_processing:
            torch.save(self.generator.module.state_dict(), generator_weight_path)
            torch.save(self.discriminator.module.state_dict(), discriminator_weight_path)
        else:
            torch.save(self.generator.state_dict(), generator_weight_path)
            torch.save(self.discriminator.state_dict(), discriminator_weight_path)

