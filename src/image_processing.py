#! /usr/bin/env python3
# -*- Coding: utf-8 -*-

import numpy as np
import cv2
import math


def modcrop(img, scale):
    if img.ndim == 3:
        img_size = np.array(img.shape[0:2])
        img_size = img_size - np.mod(img_size, scale)
        out_img = img[0:img_size[0], 0: img_size[1], :]
    else:
        img_size = np.array(img.shape[0:2])
        img_size = img_size - np.mod(img_size, scale)
        out_img = img[0:img_size[0], 0: img_size[1]]

    return out_img


def calc_psnr(ref, img):
    if ((0 <= ref).all() & (ref <= 255).all() & (0 <= img).all() & (img <= 255).all()):
        ref = ref.astype(np.float64)
        img = img.astype(np.float64)

        mse = np.mean((ref - img) ** 2, dtype=np.float64)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    else:
        print('output image is not from 0 to 255')


def image_shave(img: np.ndarray, shave_x: int, shave_y: int):
    return img[shave_x: -shave_x,
               shave_y: -shave_y]


def tensor2cvimage(tensor):
    # cf) torchvision.utils.save_image
    tensor = tensor.squeeze(0)
    ndarr = tensor.mul_(255).add_(0.5).clamp_(
        0, 255).permute(
        1, 2, 0).to(
        'cpu', torch.uint8).numpy()
    return ndarr


def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2YCrCb)
    im_ycbcr = im_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0]*(235-16)+16)/255.0  # to [16/255, 235/255]
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:]*(240-16)+16)/255.0  # to [16/255, 240/255]
    return im_ycbcr


def ycbcr2rgb(im_ycbcr):
    im_ycbcr = im_ycbcr.astype(np.float32)
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0]*255.0-16)/(235-16)  # to [0, 1]
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:]*255.0-16)/(240-16)  # to [0, 1]
    im_ycrcb = im_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCrCb2RGB)
    return im_rgb
