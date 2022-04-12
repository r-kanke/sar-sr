import torch
import torch.nn as nn

class EdgeExtractor(nn.Module):
    def __init__(self, opt, in_channel=1):
        super().__init__()

        if opt['kernel_size'] != 3:
            raise ValueError('only kernel size 3 is supported! but {}'.format(opt['kernel_size']))

        self.in_channel = in_channel
        self.sobel = opt['sobel']
        self.laplacian = opt['laplacian']
        self.log = opt['log']
        self.n_filter = [self.sobel, self.sobel, self.laplacian, self.log].count(True)

        def filter_3x3(weight_list, in_channel):
            kernel = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                            kernel_size=3, stride=1, padding=1)
            kernel.state_dict()['weight'][0] = torch.FloatTensor(weight_list)  #REVIEW: which is better? kernel.state_dict()['weight'][0] = torch.cuda.FloatTensor(weight_list) if torch.cuda.is_available() else torch.FloatTensor(weight_list)
            kernel.state_dict()['bias'].zero_()
            kernel.requires_grad_(False)
            return kernel

        modules = {}
        # sobel filter
        if self.sobel:
            weight_sobel_h = [[1, 2, 1],
                              [0, 0, 0],
                              [-1, -2, -1]]
            weight_sobel_w = [[1, 0, -1],
                              [2, 0, -2],
                              [1, 0, -1]]
            modules['sobel_h'] = filter_3x3(weight_list=weight_sobel_h, in_channel=in_channel)
            modules['sobel_w'] = filter_3x3(weight_list=weight_sobel_w, in_channel=in_channel)

        # laplacian filter
        if self.laplacian:
            weight_laplacian = [[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]]
            modules['laplacian'] = filter_3x3(weight_list=weight_laplacian, in_channel=in_channel)

        # laplacian of gaussian
        if opt['log']:
            raise NotImplementedError('LoG filter is not implemented yet!')

        self.edge_kernels = nn.ModuleDict(modules)

    def forward(self, im):
        edges = []

        if self.sobel:
            edges.append(self.edge_kernels['sobel_h'](im))
            edges.append(self.edge_kernels['sobel_w'](im))

        if self.laplacian:
            edges.append(self.edge_kernels['laplacian'](im))

        return edges

    def __len__(self):
        return self.n_filter


class Discriminator(nn.Module):
    """Discriminator of GAN.
    """
    def __init__(self, input_shape, edge_extractor_opt, patch_gan=True):
        super().__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w) if patch_gan else (1,)

        # edge extractor
        self.edge = EdgeExtractor(opt=edge_extractor_opt, in_channel=1)
        if len(self.edge) != 0:
            in_channels += self.edge.n_filter
        else:
            self.edge = None

        # network
        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(in_channels=out_filters, out_channels=out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        if patch_gan:
            # state size. 512 x 16 x 16 => 1 x 16 x 16
            layers.append(nn.Conv2d(in_channels=out_filters, out_channels=1, kernel_size=3, stride=1, padding=1))
        else:
            # state size. 512 x 16 x 16 => 1024 => 1
            layers.append(nn.Flatten())
            layers.append(nn.Linear(in_features=512 * patch_h * patch_w, out_features=1024))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Linear(in_features=1024, out_features=1))

        self.model = nn.Sequential(*layers)

    def forward(self, im):
        # add edge image as new channel
        if self.edge is not None:
            edges = self.edge(im)  # list of (B,1,H,W)
            im = torch.cat((im, *edges), dim=1)  # (B,1,H,W) -> (B,3,H,W) : (org, sobel_h, sobel_w)

        return self.model(im)


