import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from . import rcan
from . import scan
from . import discriminator

def cal_gradient_penalty(discriminator, x, x_fake, cuda=False):
    """
    gradient penaltyを計算する

    Parameters
    ----------
    model : EDModel or CodeDiscriminator
        gradient penaltyを適用したい識別器(D)またはコード識別器(CD)のインスタンス
    x : torch.Tensor
        訓練データ
    x_fake : torch.Tensor
        生成器による生成データ
    cuda : bool
        computate on gpu or not.

    Returns
    -------
    penalty : float
        gradient penaltyの値
    """
    # xと同じ次元を作成
    alpha_size = tuple((len(x), *(1, ) * (x.dim() - 1)))
    if cuda:
        alpha = torch.cuda.FloatTensor(*alpha_size).to('cuda').uniform_() 
    else:
        alpha = torch.Tensor(*alpha_size).to('cpu').uniform_()
    # ランダムな係数で補間する
    x_hat = (x.data * alpha + x_fake.data * (1 - alpha)).requires_grad_()

    def eps_norm(x):
        """
        L2ノルムを計算する.eps=1e-15.

        Parameters
        ----------
        x : torch.Tensor
            入力データ

        Returns
        -------
        torch.Tensor
            入力のL2ノルム
        """
        x = x.view(len(x), -1)
        return (x * x + 1e-15).sum(-1).sqrt()

    def bi_penalty(x):
        """
        入力と1との二乗誤差を計算する

        Parameters
        ----------
        x : torch.Tensor
            入力データ

        Returns
        -------
        torch.Tensor
            計算された二乗誤差
        """
        return (x - 1)**2

    # x_hatに関するDの勾配を計算
    grad_x_hat = torch.autograd.grad(discriminator(x_hat).sum(),
                                     x_hat,
                                     create_graph=True,
                                     only_inputs=True)[0]

    # 勾配のnormを1にするためのペナルティ項を計算
    penalty = bi_penalty(eps_norm(grad_x_hat)).mean()
    return penalty


class WGAN_GP(nn.Module):
    def __init__(self, opt, cal_psnr_ssim, cal_gradient_penalty):
        super().__init__()

        self.loss_ratio = opt.loss_ratio
        self.cal_psnr_ssim = cal_psnr_ssim
        self.cal_gradient_penalty = cal_gradient_penalty
        self.s1_data_range = opt.s1_data_range

        self.parallel_processing = opt.parallel_processing
        self.device = opt.device

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

        self.discriminator = discriminator.Discriminator(
            input_shape=opt.out_shape, edge_extractor_opt=opt.edge_extractor_opt, patch_gan=False).to(opt.device)

        # parallel processing
        if self.parallel_processing:
            self.generator = torch.nn.DataParallel(self.generator)
            self.discriminator = torch.nn.DataParallel(self.discriminator)

        # loss function
        self.criterion_pixel = nn.L1Loss().to(opt.device)

        # optimizer
        params_g = self.generator.module.parameters() if self.parallel_processing else self.generator.parameters()
        params_d = self.discriminator.module.parameters() if self.parallel_processing else self.generator.parameters()
        self.optim_g = torch.optim.Adam(params=params_g, lr=opt.lr_g, betas=(opt.b1, opt.b2))
        self.optim_d = torch.optim.Adam(params=params_d, lr=opt.lr_d, betas=(opt.b1, opt.b2))

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
        real_lr = torch.autograd.Variable(torch.cat((batch.lows1, batch.lows2), 1).type(dtype=self.Tensor, non_blocking=True))
        real_hr = torch.autograd.Variable(batch.highs1.type(dtype=self.Tensor, non_blocking=True))

        # set train mode
        self.freeze_params(self.generator, self.discriminator)
        self.unfreeze_params(self.generator)

        # forward prop
        fake_hr = self.generator(real_lr)

        # pixel loss (L1 loss)
        loss_pixel = self.criterion_pixel(fake_hr, real_hr)

        # adv loss (Wasserstein distance)
        fake_validity = self.discriminator(fake_hr).mean()
        loss_adv = -fake_validity

        loss_g = self.loss_ratio['pixel'] * loss_pixel + self.loss_ratio['adv'] * loss_adv

        # back prop
        self.optim_g.zero_grad()
        loss_g.backward()
        self.optim_g.step()
        self.scheduler_g.step()

        # calc psnr and ssim
        psnr, ssim = self.cal_psnr_ssim(fake_hr, real_hr, data_range=self.s1_data_range)

        return {'train_loss_g': loss_pixel.item(), 'train_loss_pixel': loss_pixel.item(), 'train_loss_adv': loss_adv.item(),
                'train_psnr': psnr, 'train_ssim': ssim, 'lr_g': self.scheduler_g.get_last_lr()[0],}

    def train_d(self, batch):
        """Train discriminator using one mini-batch images.

        Args:
            batch (Tensor): [description]
        """
        real_lr = torch.autograd.Variable(torch.cat((batch.lows1, batch.lows2), 1).type(dtype=self.Tensor, non_blocking=True))
        real_hr = torch.autograd.Variable(batch.highs1.type(dtype=self.Tensor, non_blocking=True))

        # set train mode
        self.freeze_params(self.generator, self.discriminator)
        self.unfreeze_params(self.discriminator)

        # forward prop
        fake_hr = self.generator(real_lr)

        # discriminator loss
        real_validity = self.discriminator(real_hr).mean()
        fake_validity = self.discriminator(fake_hr.detach()).mean()
        gradient_penalty = self.cal_gradient_penalty(self.discriminator, real_hr, fake_hr, cuda=(self.device == torch.device('cuda')))
        loss_d = -real_validity + fake_validity + self.loss_ratio['gp'] * gradient_penalty

        # backporp
        self.optim_d.zero_grad()
        loss_d.backward()
        self.optim_d.step()
        self.scheduler_d.step()

        # save result
        train_info = {'train_loss_d': loss_d.item(), 'train_real_valid': real_validity.item(),
                      'train_fake_valid': fake_validity.item(), 'train_gp': gradient_penalty.item(),
                      'lr_d': self.scheduler_d.get_last_lr()[0],}

        return train_info

    def validate(self, loader, verbose=True):
        val_info = {'val_loss_d': 0., 'val_loss_g': 0., 'val_loss_pixel': 0., 'val_loss_adv': 0.,
                    'val_valid_real': 0., 'val_valid_fake': 0., 'val_gp': 0.,
                    'val_psnr': 0., 'val_ssim': 0.,}
        
        self.freeze_params(self.generator, self.discriminator)
        self.unfreeze_params()
        
        for i, batch in enumerate(loader):
            if verbose:
                print('\r   === validating now [{}/{}] ==='.format(i+1, len(loader)), end='')

            info_g = self.validate_g(batch)
            info_d = self.validate_d(batch)
            
            for k, v in info_g.items():
                val_info[k] += v / len(loader)
            
            for k, v in info_d.items():
                val_info[k] += v / len(loader)

        return val_info

    def validate_g(self, batch):
        """Train generator only using pixel loss.

        Train using one mini-batch images.

        Args:
            batch (Tensor): [description]
        """
        # preproces
        real_lr = torch.autograd.Variable(torch.cat((batch.lows1, batch.lows2), 1).type(dtype=self.Tensor, non_blocking=True))
        real_hr = torch.autograd.Variable(batch.highs1.type(dtype=self.Tensor, non_blocking=True))

        # set eval mode
        self.freeze_params(self.generator, self.discriminator)
        self.unfreeze_params()

        # forward prop
        fake_hr = self.generator(real_lr)

        # pixel loss (L1 loss)
        loss_pixel = self.criterion_pixel(fake_hr, real_hr)

        # adv loss (Wasserstein distance)
        fake_validity = self.discriminator(fake_hr).mean()
        loss_adv = -fake_validity

        loss_g = self.loss_ratio['pixel'] * loss_pixel + self.loss_ratio['adv'] * loss_adv

        # calc psnr and ssim
        psnr, ssim = self.cal_psnr_ssim(fake_hr, real_hr, data_range=self.s1_data_range)

        return {'val_loss_g': loss_g.item(), 'val_loss_pixel': loss_pixel.item(), 'val_loss_adv': loss_adv.item(),
                'val_psnr': psnr, 'val_ssim': ssim,}

    def validate_d(self, batch):
        """Validate discriminator using one mini-batch images.

        Args:
            batch (Tensor): [description]
        """
        real_lr = torch.autograd.Variable(torch.cat((batch.lows1, batch.lows2), 1).type(dtype=self.Tensor, non_blocking=True))
        real_hr = torch.autograd.Variable(batch.highs1.type(dtype=self.Tensor, non_blocking=True))

        # set eval mode
        self.freeze_params(self.generator, self.discriminator)
        self.unfreeze_params()

        # forward prop
        fake_hr = self.generator(real_lr)

        # discriminator loss
        real_validity = self.discriminator(real_hr).mean()
        fake_validity = self.discriminator(fake_hr.detach()).mean()
        gradient_penalty = self.cal_gradient_penalty(self.discriminator, real_hr, fake_hr, cuda=(self.device == torch.device('cuda')))
        loss_d = -real_validity + fake_validity + self.loss_ratio['gp'] * gradient_penalty

        # save result
        val_info = {'val_loss_d': loss_d.item(), 'val_valid_real': real_validity.item(),
                    'val_valid_fake': fake_validity.item(), 'val_gp': gradient_penalty.item(),}

        return val_info

    def freeze_params(*args):
        for module in args:
            if module:
                for p in module.parameters():
                    p.requires_grad = False
    
    def unfreeze_params(*args):
        for module in args:
            if module:
                for p in module.parameters():
                    p.requires_grad = True

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


