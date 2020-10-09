import numpy as np
import cv2
import math
import apporchid.agora.cv.table.table_image_splitter as table_image_splitter
import apporchid.agora.cv.line.line_detection as line_detection
import apporchid.agora.cv.line.line_properties as line_properties
from apporchid.common.utils import ndarray_to_list
from apporchid.common.config import cfg as cfg
import apporchid.agora.cv.image.image_operations as image_operations
import operator
import random
from scipy import stats

debug_flag = cfg['debug']
debug_dir = cfg["debug-dir"]
output_dir = cfg['fujitsu']['medical-receipt']['output-dir']


def tabular_image_deskew(img, hor_scale, vert_scale, contour_level, file_name):
    img_cpy = img.copy()
    area_img = img.shape[0] * img.shape[1]
    table_rois, bboxline_coords, bboxs = table_image_splitter.split_image_2_table_images(img_cpy, hor_scale,
                                                                                         vert_scale,contour_level,
                                                                                         file_name)
    max_angle = 0
    angles = []
    for roi in table_rois:
        roi = roi.astype(np.uint8)
        coords = np.column_stack(np.where(roi > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if 0 < abs(angle) < 90:
            pass
        else:
            continue

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        angles.append(angle)

    if(len(angles) == 0):
        rotated, max_angle = tabular_image_deskew_edgecases(img, hor_scale, vert_scale, contour_level, file_name)
        return rotated, max_angle
    else:
        max_angle = np.median(angles)
        (h, w) = img_cpy.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, max_angle, 1.0)
        rotated = cv2.warpAffine(img_cpy, M, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(output_dir + file_name + '_rotated.jpg', rotated)
        return rotated, max_angle

def get_roi_from_bboxs(bw,bboxs):
    table_rois = []
    for bbox in bboxs:
        x,y,w,h = bbox
        snip_input = np.zeros(bw.shape, dtype = int)
        snip_input = snip_input.astype('uint8')
        snip_input[y:y+h+1, x:x+w+1] = bw[y:y+h+1, x:x+w+1]
        table_rois.append(snip_input.copy())
    return table_rois

def tabular_image_deskew_edgecases(img, hor_scale, vert_scale, contour_level, file_name):
    img_cpy = img.copy()
    area_img = img.shape[0] * img.shape[1]
    gray_image = image_operations.convert_to_gray(img)
    bw = cv2.adaptiveThreshold(~gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    bboxs = table_image_splitter.get_all_rect_bbox(bw, hor_scale, vert_scale, contour_level, file_name)
    table_rois = get_roi_from_bboxs(bw, bboxs)
    max_angle = 0
    angles = []
    for roi in table_rois:
        roi = roi.astype(np.uint8)
        coords = np.column_stack(np.where(roi > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if 0 < abs(angle) < 90:
            pass
        else:
            continue

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        angles.append(angle)
    if(len(angles) == 0):
        max_angle = 0
        cv2.imwrite(output_dir + file_name + '_rotated.jpg', img_cpy)
        return img_cpy, max_angle
    else:
        max_angle = np.median(angles)
        (h, w) = img_cpy.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, max_angle, 1.0)
        rotated = cv2.warpAffine(img_cpy, M, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(output_dir + file_name + '_rotated.jpg', rotated)
        return rotated, max_angle

def tabular_image_deskew_hough(img, file_name):
    img_b = image_operations.convert_to_binary(img)
    img_b = image_operations.detect_horizontal(img_b, scale=100, excess_size=0)
    lines = cv2.HoughLines(img_b, 1, np.pi / 180, 500, None, 0, 0)
    lines = ndarray_to_list(lines)
    if lines is not None:
        common_theta = []
        for i in range(0, len(lines)):
            # print(lines[1])
            rho = lines[i][0]
            theta = lines[i][1]
            common_theta.append(theta)
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

        most_common_angle = stats.mode(common_theta)[0][0]
        angle_in_degree = most_common_angle * (180 / math.pi)
        angle_in_degree = angle_in_degree - 90
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_in_degree, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        cv2.imwrite(output_dir + file_name + '_rotated.jpg', rotated)
        return rotated
    else:
        return img.copy()

def deg2rad(angle):
    theta = angle * (np.pi/180)
    return theta

def rad2deg(angle):
    theta = angle * (180/np.pi)
    return theta

def tabular_image_deskew_fft(img, file_name):
    angle = 0
    img_bin = image_operations.convert_to_binary(img)
    f = np.fft.fft2(img_bin)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    phase_spectrum = np.angle(fshift)

    magnitude_spectrum = magnitude_spectrum.astype('uint8')
    magnitude_spectrum_bin = cv2.adaptiveThreshold(~magnitude_spectrum, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                   cv2.THRESH_BINARY, 15, -2)
    magnitude_spectrum_bin_filtered = cv2.medianBlur(magnitude_spectrum_bin, 15)

    img_copy = img.copy()
    lines = cv2.HoughLines(magnitude_spectrum_bin_filtered, 1, np.pi / 180, 200)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

    if theta < deg2rad(45):
        angle = rad2deg(theta)
    elif theta < deg2rad(90):
        angle = rad2deg(((np.pi / 2) - theta))
    elif theta < deg2rad(135):
        angle = rad2deg(((np.pi / 2) - theta))
    else:
        angle = -rad2deg((np.pi - theta))

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_copy, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(output_dir + file_name + '_rotated.jpg', rotated)
    return rotated, angle
