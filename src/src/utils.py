import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from . import consistency


def model_to(model, device='cpu'):
    '''
    * model: torch.nn.Module or torch.nn.parallel.data_parallel.DataParallel
    '''
    if isinstance(model, torch.nn.DataParallel):
        model = model.module.to(device)
    elif isinstance(model, torch.nn.Module):
        model = model.to(device)
    else:
        raise TypeError('model must be torch.nn.Module or'
                        ' torch.nn.parallel.data_parallel.DataParallel but : '
                        + str(type(model)))

    return model


# Calc image evaluation metrics
def cal_psnr_ssim(img1, img2, data_range=1):
    '''
        img1, img2: tensor (channel, height, width)
        data_range: range of pixel values
    '''
    dim = len(img1.size())                 # Check dimension of the image
    img1 = img1.to('cpu').detach().numpy() # Tensor to ndarray
    img2 = img2.to('cpu').detach().numpy()

    psnr = 0
    ssim = 0

    # if single image
    if dim == 3:
        # (channel, height, width) -> (height, width, channel)
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
        psnr = peak_signal_noise_ratio(
            img1, img2, data_range=data_range)
        ssim = structural_similarity(
            img1, img2, data_range=data_range, multichannel=True)

    # if batch is passed
    else:
        img1 = np.transpose(img1, (0, 2, 3, 1))
        img2 = np.transpose(img2, (0, 2, 3, 1))
        # Init
        psnr_sum = 0
        ssim_sum = 0
        n_batchs = img1.shape[0]
        for i in range(n_batchs):
            _psnr = peak_signal_noise_ratio(
                img1[i], img2[i], data_range=data_range)
            _ssim = structural_similarity(
                img1[i], img2[i], data_range=data_range, multichannel=True)
            psnr_sum += _psnr
            ssim_sum += _ssim

        # calc mean
        psnr = psnr_sum / n_batchs
        ssim = ssim_sum / n_batchs

    return psnr, ssim


def tensors_to_ndarray(batch):
    # device = batch.device
    # batch_np = batch.to('cpu').detach().numpy().copy()
    # batch = batch.to(device)
    batch_np = batch.detach().clone().to('cpu').numpy()  #REVIEW: if this line works

    im_list = []
    for im in batch_np:
        im = im.transpose(1, 2, 0)  # matplotlib expects (hight, width, channel)
        im_list.append(im)
    return im_list


def find_log_file(log_dir, timestamp, ext='.csv', verbose=True):
    log_path_regex = log_dir+'/*/*'+timestamp+'*'+ext
    if verbose:
        print('Search for:', log_path_regex)

    log_path = glob.glob(log_path_regex)
    if len(log_path) > 0:
        return log_path

    raise FileNotFoundError('{} log isn\'t found under {}'.format())


def find_model_file(timestamp, ext='.pth', verbose=True):
    # search for "model/*_timestamp*.pth"
    model_path_regex = '/home/vit134/vit/sar_sr/model/*'+timestamp+'*'+ext
    if verbose:
        print('Search for:', model_path_regex)

    model_path = sorted(glob.glob(model_path_regex))

    if len(model_path) > 0:
        if verbose:
            print('Found:\n{}'.format('\n'.join(model_path)))
        return model_path

    # search for "model/*/*_timestamp*.pth"
    model_path_regex = '/home/vit134/vit/sar_sr/model/*/*'+timestamp+'*'+ext
    if verbose:
        print('Not found\nSearch for:', model_path_regex)

    model_path = sorted(glob.glob(model_path_regex))

    if len(model_path) > 0:
        if verbose:
            print('Found:\n{}'.format('\n'.join(model_path)))
        return model_path

    # search for "model/*/*_timestamp*.pth"
    model_path_regex = '/home/vit134/vit/mnt/hdd/vit/model_tmp/*'+timestamp+'*'+ext
    if verbose:
        print('Search for:', model_path_regex)
    
    model_path = sorted(glob.glob(model_path_regex))
    
    if len(model_path) > 0:
        if verbose:
            print('Found:\n{}'.format('\n'.join(model_path)))
        return model_path

    raise FileNotFoundError('{} model isn\'t fount about {}'.format(timestamp, model_path_regex))


def show_start_time(start_timestamp, save_log, save_model, log_path, model_path):
    print('start: {}\t( {} )'.format(
        start_timestamp.strftime('%Y/%m/%d %H:%M:%S'),
        start_timestamp.strftime('%Y%m%d-%H%M%S')))
    print('log is saved to: {}'.format(log_path)
          if save_log else 'log is NOT saved')
    print('model is saved to: {}'.format(model_path)
          if save_model else 'model is NOT saved')


def show_end_time(start_timestamp, end_timestamp):
    print('end: {}'
          .format(end_timestamp.strftime('%Y/%m/%d %H:%M:%S')))

    delta = end_timestamp - start_timestamp
    m, s = divmod(delta.seconds, 60)
    h, m = divmod(m, 60)
    ms, _ = divmod(delta.microseconds, 10000)
    print(f'Took : {delta.days}d, {h}:{m}:{s}.{ms}')


def plot_ims(ims=[], labels=None, row=2, col=3, color='gray'):
    '''
    * labels: list of str. image labels
    * ims: list of ndarray. images
    '''
    if len(ims) != row*col:
        raise TypeError('invalid input shape. im[{im}] != {row}x{col} '
                        .format(im=len(ims), row=row, col=col))
    if labels is not None and len(labels) != len(ims):
        raise ValueError('length not mutch. ims:[{}] labels[{}]'
                         .format(len(ims), len(labels)))

    axes = []
    fig = plt.figure(figsize=(15,10))

    for i in range(row*col):
        axes.append(fig.add_subplot(row, col, i+1))
        if labels is not None:
            axes[-1].set_title(labels[i])
        plt.imshow(ims[i], cmap=color)

    fig.tight_layout()


def get_data_augmentation(p=0.5, choice=True):
    raise NotImplementedError


def interpolate(batch, interpolation='bicubic',
                model=None, cuda=False, use_optical=True,
                sr_ims=None):
    """Apply super resolution model to batch.

    Model is applied to batch images. It is not necessary to move model
    to cpu and set model as no_grad mode before call this function.
    They are done inside this function.

    Args:
        batch (src.sen12ms.Dataset.ImTriplet): batch of input images.
        interpolation (str, optional): interpolation method. 'sr',
            'nearest' (nearest neighbor), 'bilinear', 'bicubic'
            interpolations are supported. Defaults to 'bicubic'.
        model (tensor.nn.Module, optional): super resolution model.
            Defaults to None.
        cuda (bool, optional): Whether gpu is used. Defaults to False.
        use_optical (bool, optical):
        sr_ims (torch.Tensor, optional):

    Returns:
        torch.Tensor(B,1,H,W): interpolated (or super resolutioned)
            images.
    """
    if interpolation == 'nearest':
        y = torch.nn.functional.interpolate(batch.lows1, (256,256), mode=interpolation,)

    elif interpolation in ['bilinear', 'bicubic']:
        y = torch.nn.functional.interpolate(batch.lows1, (256,256), mode=interpolation, align_corners=False)

    elif interpolation == 'sr':
        if model is None:
            raise RuntimeError('model is None')

        device = torch.device('cuda' if cuda else 'cpu')
        dtype = torch.cuda.FloatTensor if cuda else torch.Tensor
        lr = torch.cat((batch.lows1, batch.lows2), 1) if use_optical else batch.lows1
        lr = lr.type(dtype=dtype)

        model.eval()
        model = model_to(model, device=device)

        with torch.no_grad():
            y = model(lr)
    
    elif interpolation == 'correction':  #DEBUGGING: Check if works
        if cuda:
            sr_np = sr_ims.detach().clone().cpu().numpy().astype(np.float32)
            lr_np = batch.lows1.detach().clone().cpu().numpy().astype(np.float32)
        else:
            sr_np = sr_ims.detach().clone().numpy().astype(np.float32)
            lr_np = batch.lows1.detach().clone().numpy().astype(np.float32)
        
        corrected_np = consistency.correct(hr=sr_np, lr=lr_np, correction='ave_add_uniformly')
        
        device = torch.device('cuda' if cuda else 'cpu')
        dtype = torch.cuda.FloatTensor if cuda else torch.Tensor
        y = torch.from_numpy(corrected_np).type(dtype).to(device)

    return y


def test_metrics(model, dataset,
                 shuffle=True, full_comparison=True, cuda=False,
                 use_optical=True,
                 verbose=True):
    """Calculate average of psnr and ssim about all test data.

    Args:
        model (nn.Module): test model.
        dataset (src.sen12ms.Dataset): sen12ms dataset.
        shuffle (bool, optional): whether data is used randomly.
            Defaults to True.
        full_comparison (bool, optional): whether nearest neighbor and
            bilinear interploation is processed or not. Defaults to
            False.
        cuda (bool, optional): whether process is executed on GPU.
            Defaults to False.
        use_optical (bool, optional): whether the model expects optical
            data as input. Defaults to True.
        verbose (bool, optional): whether display result message or not.
            Defaults to True.

    Returns:
        dict: key is interpolation method. Each list has two dimensions,
            which 0 is psnr, 1 is ssim value.
    """
    result = {'sr': [0, 0], 'cr': [0, 0], 'bicubic': [0, 0], 'bilinear': [0, 0], 'nearest': [0, 0]}
    data_range = 5.5820  # range of sar data

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle, num_workers=6)

    for i, batch in enumerate(loader):
        # show progress
        if verbose:
            print('\r[{}/{}]'.format(i+1, len(loader)), end='')

        im = interpolate(batch, 'bicubic')
        bicubic = cal_psnr_ssim(im, batch.highs1, data_range=data_range)
        result['bicubic'][0] += bicubic[0]  # psnr
        result['bicubic'][1] += bicubic[1]  # ssim

        sr_im = interpolate(batch, 'sr', model=model, cuda=cuda, use_optical=use_optical)
        sr = cal_psnr_ssim(sr_im, batch.highs1, data_range=data_range)
        result['sr'][0] += sr[0]  # psnr
        result['sr'][1] += sr[1]  # ssim

        if full_comparison:
            im = interpolate(batch, 'bilinear')
            bilinear = cal_psnr_ssim(im, batch.highs1, data_range=data_range)
            result['bilinear'][0] += bilinear[0]
            result['bilinear'][1] += bilinear[1]

            im = interpolate(batch, 'nearest')
            nearest = cal_psnr_ssim(im, batch.highs1, data_range=data_range)
            result['nearest'][0] += nearest[0]
            result['nearest'][1] += nearest[1]

            im = interpolate(batch, 'correction', cuda=cuda, sr_ims=sr_im)
            cr = cal_psnr_ssim(im, batch.highs1, data_range=data_range)
            result['cr'][0] += cr[0]
            result['cr'][1] += cr[1]

    # calc average of metrics
    result['sr'][0] /= len(loader)
    result['sr'][1] /= len(loader)
    result['bicubic'][0] /= len(loader)
    result['bicubic'][1] /= len(loader)

    if full_comparison:
        result['nearest'][0] /= len(loader)
        result['nearest'][1] /= len(loader)
        result['bilinear'][0] /= len(loader)
        result['bilinear'][1] /= len(loader)
        result['cr'][0] /= len(loader)
        result['cr'][1] /= len(loader)

    # show result
    if verbose:
        print('\n         PSNR    SSIM')

        if full_comparison:
            print('Nearest  {p_n:.4f} {s_n:.4f}\n'
                  'Bilinear {p_bl:.4f} {s_bl:.4f}\n'
                  'Bicubic  {p_bc:.4f} {s_bc:.4f}\n'
                  'SR       {p_sr:.4f} {s_sr:.4f}\n'
                  'CR       {p_cr:.4f} {s_cr:.4f}'
                  .format(p_n=result['nearest'][0], s_n=result['nearest'][1],
                          p_bl=result['bilinear'][0], s_bl=result['bilinear'][1],
                          p_bc=result['bicubic'][0], s_bc=result['bicubic'][1],
                          p_sr=result['sr'][0], s_sr=result['sr'][1],
                          p_cr=result['cr'][0], s_cr=result['cr'][1]
                          ))
        else:
            print('Bicubic  {p_bc:.4f} {s_bc:.4f}\n'
                  'SR       {p_sr:.4f} {s_sr:.4f}\n'
                  .format(p_bc=result['bicubic'][0], s_bc=result['bicubic'][1],
                          p_sr=result['sr'][0], s_sr=result['sr'][1],
                          ))

    return result


def apply(model, dataset, n_test, shuffle=True, full_comparison=True,
          use_optical=True, cuda=True,
          save_dir=None, show_im=True, verbose=True):
    """Apply super resolution model to some data.

    Model is applied to data and compared with other interpolation
    methods.

    Args:
        model (nn.Module): super resolution model.
        dataset (src.sen12ms.Dataset): sen12ms dataset.
        n_test (int): number of test data.
        shuffle (bool, optional): whether data is used randomly.
            Defaults to True.
        full_comparison (bool, optional): whether nearest neighbor and
            bilinear interploation is processed or not. Defaults to
            False.
        save_dir (str, optional): directory path when generated
            images are saved. If None, images are not saved. Defaults to
            None.
        show_im (bool, optional): whether display result image or not.
            Defaults to True.
        verbose (bool, optional): whether display result message or not.
            Defaults to True.
    """
    # save images path like: '/home/vit134/tmp/001_Bicubic.png'
    if save_dir is not None:
        save_path = save_dir + '/SERIALNUMBER_METHOD.png'

    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=shuffle, num_workers=2)

    count = 0

    for batch in loader:
        count += 1
        if count > n_test:
            break

        # show progress
        if verbose:
            print('\r[{}/{}]'.format(count, n_test), end='')

        # do super resolution/interpolation
        sr = interpolate(batch, 'sr', model=model, cuda=cuda, use_optical=use_optical)
        bicubic = interpolate(batch, 'bicubic')
        if full_comparison:
            bilinear = interpolate(batch, 'bilinear')
            nearest = interpolate(batch, 'nearest')

        # reset norm
        low = dataset.un_normalize(batch.lows1)
        high = dataset.un_normalize(batch.highs1)
        sr = dataset.un_normalize(sr)
        bicubic = dataset.un_normalize(bicubic)
        if full_comparison:
            bilinear = dataset.un_normalize(bilinear)
            nearest = dataset.un_normalize(nearest)

        # tensor (B,C,H,W) -> ndarray (B,H,W,C)
        low = tensors_to_ndarray(low)
        high = tensors_to_ndarray(high)
        sr = tensors_to_ndarray(sr)
        bicubic = tensors_to_ndarray(bicubic)
        if full_comparison:
            bilinear = tensors_to_ndarray(bilinear)
            nearest = tensors_to_ndarray(nearest)

        # show images
        if not full_comparison:
            ims = [low[0], bicubic[0], sr[0], high[0]]
            labels = ['LR', 'Bicubic', 'SR', 'HR']
            row = 1
            col = 4
        else:
            ims = [low[0], nearest[0], bilinear[0],
                   bicubic[0], sr[0], high[0]]
            labels = ['LR', 'Nearest Neighbor', 'Bilinear',
                      'Bicubic', 'SR', 'HR']
            row = 2
            col = 3
        if show_im:
            plot_ims(ims=ims, labels=labels, row=row, col=col)
            plt.show()

        # save images
        if save_dir is not None:
            path = save_path.replace('SERIALNUMBER', '{:04}'.format(count))

            for im, label in zip(ims, labels):
                im *= 255  # rescale from 0.~1. to 0~255
                cv2.imwrite(path.replace('METHOD', label), im)

            # resize low reso images
            low = cv2.imread(path.replace('METHOD', 'LR'))
            low_256 = cv2.resize(low, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(path.replace('METHOD', 'LR_256'), low_256)
