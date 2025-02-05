
# This code is borrowed from espnet
# https://github.com/espnet/espnet/blob/a12839fb3c/espnet/transform/spec_augment.py

import random

import numpy
from PIL import Image
from PIL.Image import BICUBIC

def time_warp(x, max_time_warp=80, inplace=False, mode="PIL"):
    """time warp for spec augment
    move random center frame by the random width ~ uniform(-window, window)
    :param numpy.ndarray x: spectrogram (time, freq)
    :param int max_time_warp: maximum time frames to warp
    :param bool inplace: overwrite x with the result
    :param str mode: "PIL" (default, fast, not differentiable) or "sparse_image_warp"
        (slow, differentiable)
    :returns numpy.ndarray: time warped spectrogram (time, freq)
    """
    #new: determine random center and window boundary from frame, and warp this image to fit og dimensions and return
    window = max_time_warp
    if mode == "PIL":
        t = x.shape[0]
        if t - window <= window:
            return x
        # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
        center = random.randrange(window, t - window)
        warped = random.randrange(center - window, center + window) + 1  # 1 ... t - 1

        left = Image.fromarray(x[:center]).resize((x.shape[1], warped), BICUBIC)
        right = Image.fromarray(x[center:]).resize((x.shape[1], t - warped), BICUBIC)
        if inplace:
            x[:warped] = left
            x[warped:] = right
            return x
        return numpy.concatenate((left, right), 0)
    elif mode == "sparse_image_warp":
        import torch

        from espnet.utils import spec_augment

        # TODO(karita): make this differentiable again
        return spec_augment.time_warp(torch.from_numpy(x), window).numpy()
    else:
        raise NotImplementedError(
            "unknown resize mode: "
            + mode
            + ", choose one from (PIL, sparse_image_warp)."
        )


def freq_mask(x, F=30, n_mask=2, replace_with_zero=True, inplace=False):
    """freq mask for spec agument
    :param numpy.ndarray x: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    #new: zeros out random freq bands
    if inplace:
        cloned = x
    else:
        cloned = x.copy()

    num_mel_channels = cloned.shape[1]
    fs = numpy.random.randint(0, F, size=(n_mask, 2))

    for f, mask_end in fs:
        f_zero = random.randrange(0, num_mel_channels - f)
        mask_end += f_zero

        # avoids randrange error if values are equal and range is empty
        if f_zero == f_zero + f:
            continue

        if replace_with_zero:
            cloned[:, f_zero:mask_end] = 0
        else:
            cloned[:, f_zero:mask_end] = cloned.mean()
    return cloned

def time_mask(spec, T=40, n_mask=2, replace_with_zero=True, inplace=False):
    """freq mask for spec agument
    :param numpy.ndarray spec: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    #new: zeros out random time bands
    if inplace:
        cloned = spec
    else:
        cloned = spec.copy()
    len_spectro = cloned.shape[0]
    if isinstance(T, float):
        T = max(int(len_spectro * T), 1)  
    ts = numpy.random.randint(0, T, size=(n_mask, 2))

    for t, mask_end in ts:
        # avoid randint range error
        if len_spectro - t <= 0:
            continue
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if t_zero == t_zero + t:
            continue

        mask_end += t_zero
        if replace_with_zero:
            cloned[t_zero:mask_end] = 0
        else:
            cloned[t_zero:mask_end] = cloned.mean()
    return cloned


def spec_aug(x, args):
    """spec agument
    apply random time warping and time/freq masking
    default setting is based on LD (Librispeech double) in Table 2
        https://arxiv.org/pdf/1904.08779.pdf
    :param numpy.ndarray x: (time, freq)
    :param str resize_mode: "PIL" (fast, nondifferentiable) or "sparse_image_warp"
        (slow, differentiable)
    :param int max_time_warp: maximum frames to warp the center frame in spectrogram (W)
    :param int freq_mask_width: maximum width of the random freq mask (F)
    :param int n_freq_mask: the number of the random freq mask (m_F)
    :param int time_mask_width: maximum width of the random time mask (T)
    :param int n_time_mask: the number of the random time mask (m_T)
    :param bool inplace: overwrite intermediate array
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    assert isinstance(x, numpy.ndarray)
    assert x.ndim == 2
    if args.use_time_warp:
        x = time_warp(x, args.max_time_warp, inplace=args.inplace, mode=args.resize_mode)
    
    x = freq_mask(
        x,
        args.max_freq_width,
        args.n_freq_mask,
        inplace=args.inplace,
        replace_with_zero=args.replace_with_zero,
    )
    x = time_mask(
        x,
        args.max_time_width,
        args.n_time_mask,
        inplace=args.inplace,
        replace_with_zero=args.replace_with_zero,
    )
    return x


