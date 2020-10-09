import cv2
import numpy as np
import pandas as pd
from apporchid.common.config import cfg
import copy
import json
from apporchid.common.config import cfg as cfg

debug_dir = cfg["debug-dir"]
debug_flag = cfg['debug']

def remove_shades(img, normalize=True, fname_img='file1'):

    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((15, 15), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 3)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = np.zeros(diff_img.shape)
        norm_img = cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    try:
        if normalize:
            result = cv2.merge(result_norm_planes)
        else:
            result = cv2.merge(result_planes)
    except Exception as e:
        raise e
    if cfg['debug'] or 1 == 1:
            cv2.imwrite(cfg['debug-dir'] + fname_img + '_ShadeRemoval.png', result)
    return result

def delete_pixes(characters_boxes, img):
    img_del_pixels = copy.deepcopy(img)
    for _, box in characters_boxes.iterrows():
        for i in range(int(box.left) + 2, int(box.right) - 2):
            for j in range(int(box.top + 2), int(box.bottom) - 2):
                try:
                    img_del_pixels[j][i] = 255
                except(IndexError):
                    pass
    return img_del_pixels

