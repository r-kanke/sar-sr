"""Learning settings and hyperparameters

This module define learning settings and hyperparameres for training
networks in the SAR Super Resolution project. This module should be
imported only from 'main.ipynb.'
"""
import os
import torch

class Option():
    """Data container of learning settings and hyperparameters"""

    def __init__(self):
        """Initializer of Option

        Note:
            timestamp, log_path, model_path are not initialized.
            Don't forget to assign into them before start learning.
        """
        # args
        self.model_name = ['srgan', 'rcan', 'scan'][0]
        self.scale = 4
        self.gan_mode = [None, 'gan', 'wgan_gp'][1]

        # Paths
        self.pjt_dir = '/home/vit134/vit/sar_sr'
        self.train_dir = '/home/vit134/vit/sar_sr/data/processed/sen12ms_non_overlap_split_da/train'
        self.val_dir = '/home/vit134/vit/sar_sr/data/processed/sen12ms_non_overlap_split/validation'
        self.test_dir = '/home/vit134/vit/sar_sr/data/processed/sen12ms_non_overlap_split/test'
        self.feature_extractor_weight_path = self.pjt_dir + '/data/feature_extractor/vgg19_54.pth'
        self.log_dir = self.pjt_dir + '/log/tmp'
        self.model_dir = '/home/vit134/vit/mnt/hdd/vit/model_tmp'
        self.log_path_org = self.log_dir + '/' + self.model_name + '_' + str(self.scale) + 'x_.csv'  #TODO: delete this not-used variable
        self.model_path_org = self.model_dir + '/' + self.model_name + '_' + str(self.scale) + 'x_.pth'

        # Execution flags and settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.parallel_processing = True
        self.num_workers = 6
        self.load_lows2 = None
        self.load_highs2 = False
        self.data_augmentation = False
        self.save_log = True
        self.save_model = True
        self.save_model_interval = 250 if self.gan_mode == 'gan' else 500
        self.validation_interval = self.save_model_interval

        self.train_g_interval = 2

        # Network settings
        self.out_height = 256
        self.out_width = 256
        self.n_out_colors = 1
        self.out_shape = (self.n_out_colors, self.out_height, self.out_width)
        self.in_height = self.out_height // self.scale
        self.in_width = self.out_width // self.scale
        self.n_colors = 4  # 1, 4
        self.in_shape = (self.n_colors, self.in_height, self.in_width)

        if self.model_name == 'srgan':
            pass

        elif self.model_name == 'rcan':
            self.n_resgroups = 10
            self.n_resblocks = 20
            self.n_feats = 64

        elif self.model_name == 'scan':
            self.base_n_resgroups = 10
            self.base_n_resblocks = 20
            self.base_n_feats = 64
            self.branch_n_resblocks = 6
            self.branch_n_feats = 64
            self.branch_n_feats_sa_mid = 64

        if self.gan_mode is not None:
            self.edge_extractor_opt = {
                'kernel_size': 3,
                'sobel': False,
                'laplacian': False, 'laplacian_mode': 4,
                'log': False, 'log_sigma': 2,
                }

        # Learning hyper-parameters
        self.batch_size = 16 if self.gan_mode == 'gan' else 8

        if self.gan_mode is None:
            self.epoch = 200
            self.lr = 0.0001
            self.lr_gamma = 0.5
            self.lr_milestones = [30, 50, 100]
        else:
            self.total_steps = 300 * 1000
            self.pretrain_steps = [-1]  # [-1], [50 * 1000, 1 * 1000]
            self.lr_g = 0.0001
            self.lr_d = 0.00001 if self.gan_mode == 'gan' else 0.0001
            self.lr_gamma = 0.5  # 0.5: 3 steps from 1e-4 to 1.25e-5, 0.9: 22 steps to 1.09e-5
            self.lr_milestones_g = []
            self.lr_milestones_d = []
            self.b1 = 0.9
            self.b2 = 0.999
            self.loss_ratio = {'pixel': 1., 'content': 0, 'adv': 0.00001, 'gp': 10}

        # Dataset
        self.n_train_im = None
        self.n_val_im = None
        self.dataset_seed = 0
        self.sen12ms_mean = (0.2511, 0.2666, 0.2286, 0.5292)  # mean is round upped at 1e^-5
        self.sen12ms_std = (0.1891451, 0.1897463, 0.1905298, 0.1691035)
        self.s1_data_range = 5.5820  # [(0 - mean) / std, (1 - mean) / std] <- [0, 1]

        # datetime object for log/model file name
        self.timestamp = None
        self.log_path = None
        self.model_path = None
        self.tb_log_dir = None

        self.load_lows2 = ((self.n_colors == 4) or self.load_highs2)

    def get_options_str(self, separation='-'*60):
        """Get options as string list

        Args:
            separation (str, optional): Separation for easy-read.
            Defaults to
            '--------------------------------------------------------'.

        Returns:
            list(str): options containing some separations.

        Examples:
            To display all options:
            >>> print('\n'.join(opt.get_options_str()))
            To get csv file header:
            >>> opt.get_options_str(separation='-')
        """
        options = []

        options.append(separation)
        options.append(' pjt_dir\t' + self.pjt_dir)
        options.append(' model_name\t' + self.model_name)
        options.append(' scale\t\t' + str(self.scale))
        options.append(' gan_mode\t' + str(self.gan_mode))

        options.append(separation)
        options.append(' train_dir\t' + self.train_dir)
        options.append(' val_dir\t' + self.val_dir)
        options.append(' test_dir\t' + self.test_dir)
        options.append(' feature_extractor_weight_path\n\t\t' + self.feature_extractor_weight_path)
        options.append(' log_dir\t' + self.log_dir)
        options.append(' model_dir\t' + self.model_dir)
        options.append(' log_path_org\t' + self.log_path_org)
        options.append(' model_path_org\t' + self.model_path_org)

        options.append(separation)
        options.append(' device\t\t\t' + str(self.device))
        options.append(' parallel_processing\t' + str(self.parallel_processing))
        options.append(' num_workers\t\t' + str(self.num_workers))
        options.append(' load_highs2\t\t' + str(self.load_highs2))
        options.append(' data_augmentation\t' + str(self.data_augmentation))
        options.append(' save_log\t\t' + str(self.save_log))
        options.append(' save_model\t\t' + str(self.save_model))
        options.append(' save_model_interval\t' + str(self.save_model_interval))
        options.append(' validation_interval\t' + str(self.validation_interval))

        options.append(' train_g_interval\t' + str(self.train_g_interval))

        options.append(separation)
        options.append(' out_shape\t\t(n_out_colors, out_height, out_width)\n\t\t\t' + str(self.out_shape))
        options.append(' in_shape\t\t(n_colors, in_height, in_width)\n\t\t\t' + str(self.in_shape))

        if self.model_name == 'srresnet':
            options.append(' n_resblocks\t\t' + str(self.n_resblocks))
            options.append(' n_feats\t\t' + str(self.n_feats))
        elif self.model_name == 'rcan':
            options.append(' n_resgroups\t\t' + str(self.n_resgroups))
            options.append(' n_resblocks\t\t' + str(self.n_resblocks))
            options.append(' n_feats\t\t' + str(self.n_feats))
        elif self.model_name == 'scan':
            options.append(' base_n_resgroups\t' + str(self.base_n_resgroups))
            options.append(' base_n_resblocks\t' + str(self.base_n_resblocks))
            options.append(' base_n_feats\t\t' + str(self.base_n_feats))
            options.append(' branch_n_resblocks\t' + str(self.branch_n_resblocks))
            options.append(' branch_n_feats\t\t' + str(self.branch_n_feats))
            options.append(' branch_n_feats_sa_mid\t' + str(self.branch_n_feats_sa_mid))

        if self.gan_mode is not None:
            options.append(' edge_extractor_opt\t' + str(self.edge_extractor_opt))

        options.append(separation)

        if self.gan_mode is None:
            options.append(' batch_size\t' + str(self.batch_size))
            options.append(' epoch\t\t' + str(self.epoch))
            options.append(' lr\t\t' + str(self.lr))
            options.append(' lr_gamma\t' + str(self.lr_gamma))
            options.append(' lr_milestones\t' + str(self.lr_milestones))
        else:
            options.append(' batch_size\t\t' + str(self.batch_size))
            options.append(' total_steps\t\t' + format(self.total_steps, ','))
            options.append(' pretrain_steps\t\t' + str([format(s, ',') for s in self.pretrain_steps]))
            options.append(' lr_g\t\t\t' + format(self.lr_g, '.1e'))
            options.append(' lr_d\t\t\t' + format(self.lr_d, '.1e'))
            options.append(' lr_gamma\t\t' + str(self.lr_gamma))
            options.append(' b1\t\t\t' + str(self.b1))
            options.append(' b2\t\t\t' + str(self.b2))
            options.append(' loss_ratio\t\t' + str(self.loss_ratio))

        options.append(separation)
        options.append(' n_train_im\t' + str(self.n_train_im))
        options.append(' n_val_im\t' + str(self.n_val_im))
        options.append(' dataset_seed\t' + str(self.dataset_seed))
        options.append(' sen12ms mean\t' + str(self.sen12ms_mean))
        options.append(' sen12ms std\t' + str(self.sen12ms_std))
        options.append(' s1_data_range\t' + str(self.s1_data_range))

        options.append(separation)
        options.append(' timestamp\t' + str(self.timestamp))
        options.append(' log_path\t' + (self.log_path if isinstance(self.log_path, str) else str(self.log_path)))
        options.append(' model_path\t' + (self.model_path if isinstance(self.model_path, str) else str(self.model_path)))
        options.append(' tb_log_dir\t' + (self.tb_log_dir if isinstance(self.tb_log_dir, str) else str(self.tb_log_dir)))
        options.append(separation)

        return options

    def update_log_model_path(self, timestamp):
        """Assign log file path and model file path

        Args:
            timestamp (datetime.datetime): Timestamp of training
            start time.
        """
        self.timestamp = timestamp

        self.log_path = os.path.splitext(self.log_path_org)[0] + timestamp.strftime('%Y%m%d-%H%M%S') + os.path.splitext(self.log_path_org)[1]
        self.model_path = os.path.splitext(self.model_path_org)[0] + timestamp.strftime('%Y%m%d-%H%M%S') + os.path.splitext(self.model_path_org)[1]

        if self.gan_mode is not None:
            basename = os.path.basename(self.model_path)
            self.tb_log_dir = self.log_dir + '/' + os.path.splitext(basename)[0]
            os.makedirs(self.tb_log_dir, exist_ok=False)


if __name__ == '__main__':
    import sys
    import csv

    opt = Option()
    
    if len(sys.argv) == 1:
        print('\n'.join(opt.get_options_str()))

    elif sys.argv[1] == 'save':
        log_path = '/home/vit134/vit/sar_sr/log/tb_options.xlsx'

        data = ['']
        data += opt.get_options_str(separation='#')

        with open(log_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)
