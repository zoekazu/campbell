# /usr/env/bin python3
# encoding -*- utf-8 -*-

import numpy as np
import cv2
from src.utils import confirm_make_folder
from itertools import product
# import argparse
np.set_printoptions(threshold=np.inf)
np.random.seed(0)

DARK_START = 128
THICKNESS_START = 20
BOX_FILTER_SIZE = 3
CNN_FILTER_SIZE = 3
POOL_STEP = 4
POOL_SIZE = 2
BASE_RANGE = [118, 128]
HEIGHT_RANGE = 2
SHAVE_SIZE = 0

SEPARATE = 4

SAVE_DIR = 'make_images'


def main():
    interval = POOL_SIZE**(POOL_STEP+1)+CNN_FILTER_SIZE*POOL_STEP+2
    base_height = DARK_START * BOX_FILTER_SIZE * HEIGHT_RANGE
    base_width = sum([i+interval for i in range(1, THICKNESS_START+1)])+interval

    lines = np.zeros([base_height, base_width], dtype=np.uint8)

    width_count = 0
    for i, thickness in enumerate(reversed(range(1, THICKNESS_START+1))):
        height_count = 0
        width_count += interval

        if i == THICKNESS_START//2:
            width_separate_point = width_count

        for darkness in reversed(range(1, DARK_START+1)):
            lines[height_count:BOX_FILTER_SIZE*HEIGHT_RANGE+height_count,
                  width_count:thickness+width_count] = darkness
            height_count += BOX_FILTER_SIZE*HEIGHT_RANGE
        width_count += thickness

    if SHAVE_SIZE:
        pad_up = lines[0, :]
        pad_up = np.tile(pad_up, (SHAVE_SIZE, 1))

        pad_down = lines[-1, :]
        pad_down = np.tile(pad_down, (SHAVE_SIZE, 1))
        lines = np.concatenate([pad_up, lines, pad_down], axis=0)

        pad_side = np.zeros([lines.shape[0], SHAVE_SIZE], dtype=np.uint8)
        lines = np.concatenate([pad_side, lines, pad_side], axis=1)

    teach_before = np.where(lines != 0, 0, 255)
    lines = cv2.boxFilter(lines, -1, ksize=(BOX_FILTER_SIZE, BOX_FILTER_SIZE))

    train = np.random.randint(BASE_RANGE[0], BASE_RANGE[1], size=(
        base_height+2*SHAVE_SIZE, base_width+2*SHAVE_SIZE), dtype=np.uint8)
    train_org = train.copy()
    teach = np.ones_like(train, dtype=np.uint8)*255

    train = np.where(train >= lines, train - lines, 0)
    teach = np.where((lines != 0) | (teach_before == 0), 0, 255)

    cv2.imwrite('./{}/train_all.bmp'.format(SAVE_DIR), train)
    cv2.imwrite('./{}/teach_all.bmp'.format(SAVE_DIR), teach)
    cv2.imwrite('./{}/lines_all.bmp'.format(SAVE_DIR), lines)

    for j in range(SEPARATE):
        if (j == 0) | (j == 1):
            thickness_list = [k for k in range(THICKNESS_START//2, THICKNESS_START)]
        elif (j == 2) | (j == 3):
            thickness_list = [k for k in range(1, THICKNESS_START//2+1)]

        if (j == 0) | (j == 2):
            dark_list = [k for k in range(DARK_START//2, DARK_START)]
        elif (j == 1) | (j == 3):
            dark_list = [k for k in range(1, DARK_START//2+1)]

        interval = POOL_SIZE**(POOL_STEP+1)+CNN_FILTER_SIZE*POOL_STEP+2
        base_height = DARK_START//2 * BOX_FILTER_SIZE * HEIGHT_RANGE
        base_width = sum([i+interval for i in thickness_list])+interval

        lines = np.zeros([base_height, base_width], dtype=np.uint8)

        width_count = 0
        for i, thickness in enumerate(reversed(thickness_list)):
            height_count = 0
            width_count += interval

            for darkness in reversed(dark_list):
                lines[height_count:BOX_FILTER_SIZE*HEIGHT_RANGE+height_count,
                      width_count:thickness+width_count] = darkness
                height_count += BOX_FILTER_SIZE*HEIGHT_RANGE
            width_count += thickness

        if SHAVE_SIZE:
            pad_up = lines[0, :]
            pad_up = np.tile(pad_up, (SHAVE_SIZE, 1))

            pad_down = lines[-1, :]
            pad_down = np.tile(pad_down, (SHAVE_SIZE, 1))
            lines = np.concatenate([pad_up, lines, pad_down], axis=0)

            pad_side = np.zeros([lines.shape[0], SHAVE_SIZE], dtype=np.uint8)
            lines = np.concatenate([pad_side, lines, pad_side], axis=1)

        teach_before = np.where(lines != 0, 0, 255)
        lines = cv2.boxFilter(lines, -1, ksize=(BOX_FILTER_SIZE, BOX_FILTER_SIZE))

        train = np.random.randint(BASE_RANGE[0], BASE_RANGE[1], size=(
            base_height+2*SHAVE_SIZE, base_width+2*SHAVE_SIZE), dtype=np.uint8)
        train_org = train.copy()
        teach = np.ones_like(train, dtype=np.uint8)*255

        train = np.where(train >= lines, train - lines, 0)
        teach = np.where((lines != 0) | (teach_before == 0), 0, 255)

        cv2.imwrite('./{0}/train_part{1}.bmp'.format(SAVE_DIR, j), train)
        cv2.imwrite('./{0}/teach_part{1}.bmp'.format(SAVE_DIR, j), teach)
        cv2.imwrite('./{0}/lines_part{1}.bmp'.format(SAVE_DIR, j), lines)


if __name__ == '__main__':
    confirm_make_folder(SAVE_DIR)
    main()
