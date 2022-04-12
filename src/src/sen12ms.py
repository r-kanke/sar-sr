import os
import random
import glob
from typing import NamedTuple
from PIL import Image

import torch
import torchvision.transforms as transforms

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def norm(self, tensor):
        return self.__call__(tensor)

    def unnorm(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be unnormalized.
        Returns:
            Tensor: UnNormalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Dataset(torch.utils.data.Dataset):
    '''
    Dataset class for Sen12ms (Sentinel 1 and 2 multi-spectral) dataset.
    Original data is distributed here:
        Schmitt, Michael, et al. "SEN12MS--A Curated Dataset of
        Georeferenced Multi-Spectral Sentinel-1/2 Imagery for Deep
        Learning and Data Fusion." arXiv preprint arXiv:1906.07789
        (2019).
    NOTE: Transformation of paired images is not supported. DA must be
        Implemented before loading images from the folder.
    '''
    class ImTriplet(NamedTuple):
        lows1: torch.Tensor
        lows2: torch.Tensor
        highs1: torch.Tensor
        highs2: torch.Tensor

    class ImTripletLight(NamedTuple):
        lows1: torch.Tensor
        lows2: torch.Tensor
        highs1: torch.Tensor

    class ImPair(NamedTuple):
        lows1: torch.Tensor
        highs1: torch.Tensor

    # Called when an instace is generated: Do preprocess and getting image lists
    def __init__(self, data_dir, scale=4, highreso='gsd10',
                 transform=transforms.ToTensor(), transform_color=transforms.ToTensor(),
                 mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5),
                 load_lows2=True, load_highs2=False, n_im=None, seed=None):
        # Store args
        self.data_dir = data_dir
        self.scale = scale
        self.load_lows2 = load_lows2
        self.load_highs2 = load_highs2

        # transform
        self.transform = transform
        self.transform_color = transform_color

        # normalization
        self.norm = Normalize(mean=(mean[3],), std=(std[3],))
        self.norm_color = Normalize(mean=mean[:3], std=std[:3])

        self.highreso = highreso
        self.lowreso = self.highreso[:3] + str(int(self.highreso[3:]) * self.scale)

        # List up image list
        self.s1_paths = glob.glob(os.path.join(
            data_dir, self.highreso, 's1*', '*.png'
        ))

        # use a part of whole dataset
        if n_im is not None:
            if seed is not None:
                random.seed(seed)
            random.shuffle(sorted(self.s1_paths))
            self.s1_paths = self.s1_paths[:n_im]

        # Num of image set
        self.im_num = len(self.s1_paths)

    # get paths of low s2, high s1, high s2 images from low s1 path
    def _get_im_paths(self, highs1_path):
        highs2 = highs1_path.replace('/s1', '/s2').replace('_s1_', '_s2_')
        lows1 = highs1_path.replace(self.highreso, self.lowreso)
        lows2 = highs2.replace(self.highreso, self.lowreso)
        return (lows1, lows2, highs1_path, highs2)

    # Called by len(self) as length
    def __len__(self):
        return self.im_num

    def __getitem__(self, index):
        # Load images
        paths = self._get_im_paths(self.s1_paths[index])
        lows1 = Image.open(paths[0])
        highs1 = Image.open(paths[2])
        if self.load_lows2:
            lows2 = Image.open(paths[1])
        if self.load_highs2:
            highs2 = Image.open(paths[3])

        # transform (ToTensor)
        lows1 = self.transform(lows1)
        highs1 = self.transform(highs1)
        if self.load_lows2:
            lows2 = self.transform_color(lows2)
        if self.load_highs2:
            highs2 = self.transform_color(highs2)

        # normalize
        lows1 = self.norm(lows1)
        highs1 = self.norm(highs1)
        if self.load_lows2:
            lows2 = self.norm_color(lows2)
        if self.load_highs2:
            highs2 = self.norm_color(highs2)

        #REVIEW: this conditional branch hinder optimization
        if self.load_lows2 and not self.load_highs2:
            return Dataset.ImTripletLight(lows1=lows1, lows2=lows2, highs1=highs1)
        elif not self.load_lows2 and not self.load_highs2:
            return Dataset.ImPair(lows1=lows1, highs1=highs1)
        elif self.load_lows2 and self.load_highs2:
            return Dataset.ImTriplet(lows1=lows1, lows2=lows2, highs1=highs1, highs2=highs2)
        else:
            raise ValueError('Load full data! Why u dont load low s2 data!')

    def un_normalize(self, tensor):
        if tensor.ndim == 3:  # C, H, W
            if len(tensor) == 1:  # sar
                un_normed = self.norm.unnorm(tensor)
            elif len(tensor) == 3:  # color
                un_normed = self.norm_color.unnorm(tensor)

        elif tensor.ndim == 4:  # B, C, H, W
            un_normed = []
            for t in tensor:
                if len(t) == 1:
                    un_normed.append(self.norm.unnorm(t))
                elif len(t) == 3:
                    un_normed.append(self.norm_color.unnorm(t))
            un_normed = torch.stack(un_normed)

        return un_normed

    def random_split(self, split_ratio=0.2, seed=None):
        # fix random seed
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.s1_paths)

        # this -> this and new
        if isinstance(split_ratio, float) == 1:
            # new dataset
            split_dataset = Dataset(
                data_dir=self.data_dir, scale=self.scale, highreso=self.highreso,
                transform=self.transform, transform_color=self.transform_color,
                mean=(*self.norm_color.mean, *self.norm.mean), std=(*self.norm_color.std, *self.norm.std),
                load_highs2=self.load_highs2
                )

            # length of new dataset
            split_len = int(len(self.s1_paths) * split_ratio)

            # split
            split_dataset.s1_paths = self.s1_paths[:split_len]
            self.s1_paths = self.s1_paths[split_len:]

            # update length
            split_dataset.im_num = len(split_dataset.s1_paths)
            self.im_num = len(self.s1_paths)

            return split_dataset

        # this -> this and new1, new2
        elif len(split_ratio) == 2:
            # new datasets
            split_dataset = [
                Dataset(
                    data_dir=self.data_dir, scale=self.scale, highreso=self.highreso,
                    transform=self.transform, transform_color=self.transform_color,
                    mean=(*self.norm_color.mean, *self.norm.mean), std=(*self.norm_color.std, *self.norm.std),
                    load_highs2=self.load_highs2
                    ) for _ in range(2)
                ]

            # length of new datasets
            split_len = [int(len(self.s1_paths) * ratio) for ratio in split_ratio]

            # split
            split_dataset[0].s1_paths = self.s1_paths[: split_len[0]]
            split_dataset[1].s1_paths = self.s1_paths[split_len[0] : split_len[1]+split_len[0]]
            self.s1_paths = self.s1_paths[split_len[0]+split_len[1] :]

            # update length
            split_dataset[0].im_num = len(split_dataset[0].s1_paths)
            split_dataset[1].im_num = len(split_dataset[1].s1_paths)
            self.im_num = len(self.s1_paths)

            return split_dataset[0], split_dataset[1]

        else:
            raise ValueError('split_ratio must be float or 2 length list, but {}'.format(split_ratio))


if __name__ == '__main__':
    data_dir = '/home/vit134/vit/sar_sr/data/processed/sen12ms_non_overlap_split/train'
    batch_size = 16
    scale = 4

    # define preprocessing
    transform = transforms.Compose([transforms.ToTensor(),])
    transform_color = transforms.Compose([transforms.ToTensor(),])

    # create datasets
    dataset = Dataset(
        data_dir=data_dir,
        scale=scale,
        transform=transform,
        transform_color=transform_color
    )

    print('dataset size: {}'.format(len(dataset)))

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    for i, batch in enumerate(loader):
        print('\r[{}/{}] {}'.format(i+1, len(loader), batch.lows1.shape), end='')
    print('\n')

    # split dataset for train/val/test
    split_ratio = [0.7, 0.2]

    train_dataset, val_dataset = dataset.random_split(split_ratio=split_ratio, seed=0)
    test_dataset = dataset
    print('train: {} val: {} test: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

    for dataset in [train_dataset, val_dataset, test_dataset]:
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            )

        for i, batch in enumerate(loader):
            print('\r[{}/{}] {}'.format(i+1, len(loader), batch.lows1.shape), end='')
        print('\n')


