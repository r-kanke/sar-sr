import random

import numpy as np

def correct(hr: np.ndarray, lr: np.ndarray, correction: str = 'ave_add_randomly') -> np.ndarray:
    """Correct inconsistency between high resolution and low resolution
    images.

    Args:
        hr (np.ndarray): high resolution image. dim order must be
            (Batch, Channel, Height, Width), like tensor.
        lr (np.ndarray): high resolution image. dim order must be
            (Batch, Channel, Height // scale, Width // scale).
        correction (str, optional): specifying correction method.
            Defaults to 'ave_add_randomly'.

    Returns:
        np.ndarray: corrected image, dimension (C, H, W).
    """
    # size check
    if hr.ndim == 2:
        hr = hr.reshape(1, 1, *hr.shape)
        lr = lr.reshape(1, 1, *lr.shape)
    elif hr.ndim == 3:
        hr = hr.reshape(1, *hr.shape)
        lr = lr.reshape(1, *lr.shape)

    b, c, h, w = hr.shape
    b_lr, c_lr, h_lr, w_lr = lr.shape

    if b != b_lr:
        raise RuntimeError
    if (c != 1 and c != 3) or (c_lr != 1 and c_lr != 3):
        raise RuntimeError
    if w % w_lr != 0 or h % h_lr != 0 or w / w_lr != h / h_lr:
        raise RuntimeError

    # correction method
    if correction == 'ave_add_randomly':
        correct = _correct_ave_consistency_add_randomly
    elif correction == 'ave_add_uniformly':
        correct = _correct_ave_consistency_add_uniformly
    elif correction == 'ave_add_uniformly_uint8':
        correct = _correct_ave_consistency_add_uniformly_uint8
    else:
        raise NotImplementedError

    # correct inconsistency
    corrected = correct(hr, lr)

    return corrected


def _correct_ave_consistency_add_randomly(hr: np.ndarray, lr:np.ndarray) -> np.ndarray:
    if hr.dtype != np.uint8 or lr.dtype != np.uint8:
        raise RuntimeError

    scale = hr.shape[-1] // lr.shape[-1]
    patch_h, patch_w, patch_pixel = scale, scale, scale ** 2
    corrected = hr.copy()

    for b in range(lr.shape[0]):
        for c in range(lr.shape[1]):

            for h in range(lr.shape[2]):
                for w in range(lr.shape[3]):
                    # fetch roi patch
                    _lr = lr[b][c][h][w]

                    while True:
                        # calc consistency between lr and average of hr
                        _hr = corrected[b, c, scale * h : scale * (h + 1), scale * w : scale * (w + 1)]
                        diff = _lr * patch_pixel - _hr.sum()
                        if diff == 0: break

                        n_to_be_added = int(abs(diff) % patch_pixel)

                        # correct by adding uniformly
                        if n_to_be_added <= 0:
                            to_be_added_uniform = diff // patch_pixel

                            for h_p in range(patch_h):
                                for w_p in range(patch_w):
                                    value = corrected[b, c, h * scale + h_p, w * scale + w_p] + to_be_added_uniform
                                    if value < 0:
                                        corrected[b, c, h * scale + h_p, w * scale + w_p] = 0
                                    elif value > 255:
                                        corrected[b, c, h * scale + h_p, w * scale + w_p] = 255
                                    else:
                                        corrected[b, c, h * scale + h_p, w * scale + w_p] = value

                        # correct by adding some pixels
                        else:
                            idxs = [i for i in range(0, patch_pixel)]
                            idxs = random.sample(idxs, n_to_be_added)  #FIXME: the one pixel is rarely incremented repeatedly.

                            for i in idxs:
                                value = corrected[b, c, h * scale + i // patch_w, w * scale + i % patch_w] + (1 if diff > 0 else -1)
                                if 0 <= value and value <= 255:
                                    corrected[b, c, h * scale + i // patch_w, w * scale + i % patch_w] = value
                                    n_to_be_added -= 1

    return corrected


def _correct_ave_consistency_add_uniformly(hr: np.ndarray, lr:np.ndarray) -> np.ndarray:
    scale = hr.shape[-1] // lr.shape[-1]
    patch_pixel = scale ** 2
    corrected = hr.copy()

    for b in range(lr.shape[0]):
        for c in range(lr.shape[1]):

            for h in range(lr.shape[2]):
                for w in range(lr.shape[3]):
                    # fetch roi patch
                    _lr = lr[b][c][h][w]
                    _hr = corrected[b, c, scale * h : scale * (h + 1), scale * w : scale * (w + 1)]

                    # calc consistency between lr and average of hr
                    diff = _lr * patch_pixel - _hr.sum()

                    # correct by adding uniformly
                    to_be_added_uniform = diff / patch_pixel
                    corrected[b, c, scale * h : scale * (h + 1), scale * w : scale * (w + 1)] += to_be_added_uniform

    return corrected


def _correct_ave_consistency_add_uniformly_uint8(hr: np.ndarray, lr:np.ndarray) -> np.ndarray:
    if hr.dtype != np.uint8 or lr.dtype != np.uint8:
        raise RuntimeError

    scale = hr.shape[-1] // lr.shape[-1]
    patch_h, patch_w, patch_pixel = scale, scale, scale ** 2
    corrected = hr.copy()

    for b in range(lr.shape[0]):
        for c in range(lr.shape[1]):

            for h in range(lr.shape[2]):
                for w in range(lr.shape[3]):
                    # fetch roi patch
                    _lr = lr[b][c][h][w]

                    # calc consistency between lr and average of hr
                    _hr = corrected[b, c, scale * h : scale * (h + 1), scale * w : scale * (w + 1)]
                    diff = _lr * patch_pixel - _hr.sum()

                    # correct by adding uniformly
                    to_be_added_uniform = diff // patch_pixel

                    for h_p in range(patch_h):
                        for w_p in range(patch_w):
                            value = corrected[b, c, h * scale + h_p, w * scale + w_p] + to_be_added_uniform
                            if value < 0:
                                corrected[b, c, h * scale + h_p, w * scale + w_p] = 0
                            elif value > 255:
                                corrected[b, c, h * scale + h_p, w * scale + w_p] = 255
                            else:
                                corrected[b, c, h * scale + h_p, w * scale + w_p] = value

    return corrected


if __name__ == '__main__':
    import cv2
    import torch

    im_paths = [
        '/home/vit134/vit/sar_sr/data/sample/sr_sample/001_#INTERPOLATION#.png',
        '/home/vit134/vit/sar_sr/data/sample/sr_sample/002_#INTERPOLATION#.png',
        '/home/vit134/vit/sar_sr/data/sample/sr_sample/003_#INTERPOLATION#.png',
        '/home/vit134/vit/sar_sr/data/sample/sr_sample/004_#INTERPOLATION#.png',
        '/home/vit134/vit/sar_sr/data/sample/sr_sample/005_#INTERPOLATION#.png',
        ]

    def im_separately():
        for path in im_paths:
            sr = cv2.imread(path.replace('#INTERPOLATION#', 'SR'), cv2.IMREAD_GRAYSCALE)
            lr = cv2.imread(path.replace('#INTERPOLATION#', 'LR'), cv2.IMREAD_GRAYSCALE)

            sr_h, sr_w = sr.shape
            sr = sr.reshape(1, 1, sr_h, sr_w)
            lr_h, lr_w = lr.shape
            lr = lr.reshape(1, 1, lr_h, lr_w)

            corrected = correct(sr, lr, correction='ave_add_randomly')

            cv2.imwrite(path.replace('#INTERPOLATION#', 'CR'), corrected.reshape(sr_h, sr_w))

    def as_batch():
        sr_list = []
        lr_list = []
        for path in im_paths:
            sr = cv2.imread(path.replace('#INTERPOLATION#', 'SR'), cv2.IMREAD_GRAYSCALE)
            lr = cv2.imread(path.replace('#INTERPOLATION#', 'LR'), cv2.IMREAD_GRAYSCALE)

            sr_list.append(sr.reshape(1, *sr.shape))
            lr_list.append(lr.reshape(1, *lr.shape))

        sr = np.stack(sr_list)
        lr = np.stack(lr_list)

        cr = correct(sr, lr, correction='ave_add_randomly')

        for i, path in enumerate(im_paths):
            _cr = cr[i].reshape(cr.shape[-2], cr.shape[-1])
            cv2.imwrite(path.replace('#INTERPOLATION#', 'CR'), _cr)

    def from_tensor_as_batch():
        sr_list = []
        lr_list = []
        for path in im_paths:
            sr = cv2.imread(path.replace('#INTERPOLATION#', 'SR'), cv2.IMREAD_GRAYSCALE)
            lr = cv2.imread(path.replace('#INTERPOLATION#', 'LR'), cv2.IMREAD_GRAYSCALE)

            sr_list.append(sr.reshape(1, *sr.shape))
            lr_list.append(lr.reshape(1, *lr.shape))

        sr = np.stack(sr_list)
        lr = np.stack(lr_list)
        
        # ndarray => Tensor
        sr = torch.from_numpy(sr.astype(np.float32))
        lr = torch.from_numpy(lr.astype(np.float32))

        # Tensor => ndarray
        sr = sr.detach().clone().numpy()
        lr = lr.detach().clone().numpy()
        
        cr = correct(sr, lr, correction='ave_add_uniformly')

        for i, path in enumerate(im_paths):
            _cr = cr[i].reshape(cr.shape[-2], cr.shape[-1])
            cv2.imwrite(path.replace('#INTERPOLATION#', 'CR'), _cr)


    from_tensor_as_batch()
