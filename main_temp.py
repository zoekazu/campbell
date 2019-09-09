# /usr/env/bin python3
# encoding -*- utf-8 -*-

import numpy as np
import cv2
from src.utils import confirm_make_folder
np.set_printoptions(threshold=np.inf)

TRAIN_NUM = 100000
TEST_NUM = 1000
BOXFILTER_SIZE = 3
CONTAMINATION_SIZE_RANGE = 2


def main(base_size, base_range, cont_size, cont_range, save_dir, condition_num, phase):
    """[summary]

    Arguments:
        base_size {[type]} -- [description]
        base_range {[type]} -- [description]
        cont_size {[type]} -- [minimum size is '6']
        cont_range {[type]} -- [description]
        save_dir {[type]} -- [description]
        condition_num {[type]} -- [description]
    """

    if phase == 'train':
        img_amount = TRAIN_NUM
    elif phase == 'test':
        img_amount = TEST_NUM

    for iter in range(1, img_amount+1):
        base = np.random.randint(base_range[0], base_range[1], size=(base_size[0], base_size[1]))
        base = base.astype(np.uint8)
        base_label = np.zeros_like(base).astype(np.uint8)

        cont_size_x = np.random.randint(cont_size[0], cont_size[0]+CONTAMINATION_SIZE_RANGE)
        cont_size_y = np.random.randint(cont_size[1], cont_size[1]+CONTAMINATION_SIZE_RANGE)

        contamination = np.zeros([cont_size_x + BOXFILTER_SIZE,
                                  cont_size_y + BOXFILTER_SIZE], dtype=np.uint8)
        radius = min((cont_size_x) // 2-1, (cont_size_y) // 2-1)
        cont_luminance = np.random.randint(cont_range[0], cont_range[1])

        contamination = cv2.circle(
            contamination, ((cont_size_x + BOXFILTER_SIZE - 1) // 2, (cont_size_y + BOXFILTER_SIZE - 1) // 2), radius, (cont_luminance), -1)
        # contamination = cv2.resize(contamination, dsize=None, fx=1/2, fy=1/2, interpolation=cv2.INTER_CUBIC)
        # contamination = cv2.resize(contamination, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        contamination = cv2.boxFilter(contamination, -1, ksize=(BOXFILTER_SIZE, BOXFILTER_SIZE))
        mixing_x = np.random.randint(BOXFILTER_SIZE // 2 + 10, base_size[0] - cont_size_x - BOXFILTER_SIZE // 2 - 10)
        mixing_y = np.random.randint(BOXFILTER_SIZE // 2 + 10, base_size[1] - cont_size_y - BOXFILTER_SIZE // 2 - 10)

        base[mixing_x: mixing_x + contamination.shape[0], mixing_y: mixing_y + contamination.shape[1]] = base[mixing_x: mixing_x + contamination.shape[0],
                                                                                                              mixing_y: mixing_y + contamination.shape[1]] - contamination

        base_label[mixing_x: mixing_x + contamination.shape[0],
                   mixing_y: mixing_y + contamination.shape[1]] = contamination
        base_label = np.where(base_label == 0, 255, 0)

        cv2.imwrite('{0}/train/{1}_condition{2}_no{3}.bmp'.format(save_dir, phase, condition_num, iter), base)
        cv2.imwrite('{0}/label/{1}_condition{2}_no{3}_label.bmp'.format(save_dir,
                                                                        phase, condition_num, iter), base_label)


def run(settiing_dict, phase):
    for i, config_dict in enumerate(dict_list, start=1):
        save_dir = './images/{0}/condition{1}'.format(phase, i)
        dir_list = ['train', 'label']
        for dir_name in dir_list:
            confirm_make_folder('{0}/{1}'.format(save_dir, dir_name))
        setting_path = './images/{0}/condition{1}/config.txt'.format(phase, i)

        with open(setting_path, 'wt') as f:
            for dict_item, dict_value in config_dict.items():
                f.write(dict_item)
                f.write(str(dict_value))
                f.write('\n')

        main(**config_dict, save_dir=save_dir, condition_num=i, phase=phase)


if __name__ == '__main__':

    base_size = [128, 128]
    base_range = [220, 230]

    # radius is
    dict_list = [
        {'base_size': base_size,
         'base_range': base_range,
         'cont_size': [10, 10],
         'cont_range': [25, 30]},

        {'base_size': base_size,
         'base_range': base_range,
         'cont_size': [8, 8],
         'cont_range': [15, 25]},

        {'base_size': base_size,
         'base_range': base_range,
         'cont_size': [6, 6],
         'cont_range': [5, 10]}]

    run(dict_list, 'train')
    run(dict_list, 'test')
