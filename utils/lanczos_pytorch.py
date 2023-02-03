# Modified from:
# https://github.com/xinntao/matlab_functions_verification/blob/main/bicubic_pytorch.py
import math
import typing

import torch
from torch.nn import functional as F


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos_contribution(x, a):
    range = (x > -a) * (x < a)
    cont = sinc(x) * sinc(x/a)
    out = cont * range.to(dtype=x.dtype)
    return out / out.sum()


def get_weight(dist, kernel_size, n_order):
    buffer_pos = dist.new_zeros(kernel_size, len(dist))
    for idx, buffer_sub in enumerate(buffer_pos):
        buffer_sub.copy_(dist - idx)

    weight = lanczos_contribution(buffer_pos, n_order)
    weight /= weight.sum(dim=0, keepdim=True)
    return weight


def reflect_padding(x: torch.Tensor, dim: int, pad_pre: int,
                    pad_post: int) -> torch.Tensor:
    '''
    Apply reflect padding to the given Tensor.
    Note that it is slightly different from the PyTorch functional.pad,
    where boundary elements are used only once.
    Instead, we follow the MATLAB implementation
    which uses boundary elements twice.
    For example,
    [a, b, c, d] would become [b, a, b, c, d, c] with the PyTorch implementation,
    while our implementation yields [a, a, b, c, d, d].
    '''
    b, c, h, w = x.size()
    if dim == 2 or dim == -2:
        padding_buffer = x.new_zeros(b, c, h + pad_pre + pad_post, w)
        padding_buffer[..., pad_pre:(h + pad_pre), :].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1, :].copy_(x[..., p, :])
        for p in range(pad_post):
            padding_buffer[..., h + pad_pre + p, :].copy_(x[..., -(p + 1), :])
    else:
        padding_buffer = x.new_zeros(b, c, h, w + pad_pre + pad_post)
        padding_buffer[..., pad_pre:(w + pad_pre)].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1].copy_(x[..., p])
        for p in range(pad_post):
            padding_buffer[..., w + pad_pre + p].copy_(x[..., -(p + 1)])

    return padding_buffer


def padding(x: torch.Tensor,
            dim: int,
            pad_pre: int,
            pad_post: int,
            padding_type: str = 'reflect') -> torch.Tensor:

    if padding_type == 'reflect':
        x_pad = reflect_padding(x, dim, pad_pre, pad_post)
    else:
        raise ValueError('{} padding is not supported!'.format(padding_type))

    return x_pad


def get_padding(base: torch.Tensor, kernel_size: int,
                x_size: int) -> typing.Tuple[int, int, torch.Tensor]:

    base = base.long()
    r_min = base.min()
    r_max = base.max() + kernel_size - 1

    if r_min <= 0:
        pad_pre = -r_min
        pad_pre = pad_pre.item()
        base += pad_pre
    else:
        pad_pre = 0

    if r_max >= x_size:
        pad_post = r_max - x_size + 1
        pad_post = pad_post.item()
    else:
        pad_post = 0

    return pad_pre, pad_post, base


def reshape_tensor(x: torch.Tensor, dim: int,
                   kernel_size: int) -> torch.Tensor:
    # Resize height
    if dim == 2 or dim == -2:
        k = (kernel_size, 1)
        h_out = x.size(-2) - kernel_size + 1
        w_out = x.size(-1)
    # Resize width
    else:
        k = (1, kernel_size)
        h_out = x.size(-2)
        w_out = x.size(-1) - kernel_size + 1

    unfold = F.unfold(x, k)
    unfold = unfold.view(unfold.size(0), -1, h_out, w_out)
    return unfold


def resize_1d(x, dim, side, padding_type='reflect'):
    scale = side / x.size(dim)

    # Identity case
    if scale == 1:
        return x

    # Order number and kernel size
    if scale < 1:
        # Lanczos2 used for downscaling
        n_order = 2
    else:
        # Lanczos3 used for upscaling
        n_order = 3
    kernel_size = 2 * n_order

    # Weights only depend on the shape of input and output,
    # so we do not calculate gradients here.
    with torch.no_grad():
        d = 1 / (2 * side)
        pos = torch.linspace(
            start=d,
            end=(1 - d),
            steps=side,
            dtype=x.dtype,
            device=x.device,
        )
        pos = x.size(dim) * pos - 0.5
        base = pos.floor() - (kernel_size // 2) + 1
        dist = pos - base
        weight = get_weight(dist, kernel_size, n_order)
        pad_pre, pad_post, base = get_padding(base, kernel_size, x.size(dim))

    # To backpropagate through x
    x_pad = padding(x, dim, pad_pre, pad_post, padding_type=padding_type)
    unfold = reshape_tensor(x_pad, dim, kernel_size)
    # Subsampling first
    if dim == 2 or dim == -2:
        sample = unfold[..., base, :]
        weight = weight.view(1, kernel_size, sample.size(2), 1)
    else:
        sample = unfold[..., base]
        weight = weight.view(1, kernel_size, 1, sample.size(3))

    # Apply the kernel
    down = sample * weight
    down = down.sum(dim=1, keepdim=True)
    return down


def imresize(x, sides, padding_type='reflect'):
    assert x.dim() == 4, '{}-dim Tensor is not supported!'.format(x.dim())

    if x.dtype != torch.float32 or x.dtype != torch.float64:
        dtype = x.dtype
        x = x.float()
    else:
        dtype = None
    
    b, c, h, w = x.size()
    x = x.view(-1, 1, h, w)
    x = resize_1d(x, -2, side=sides[0], padding_type=padding_type)
    x = resize_1d(x, -1, side=sides[1], padding_type=padding_type)
    x = x.view(b, c, x.size(-2), x.size(-1))

    if dtype is not None:
        if not dtype.is_floating_point:
            x = x.round()
        if dtype is torch.uint8:
            x = x.clamp(0, 255)
        x = x.to(dtype=dtype)

    return x