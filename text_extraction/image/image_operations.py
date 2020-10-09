import cv2
import numpy as np
from apporchid.common.config import cfg as cfg
# import apporchid.agora.cv.table.table_image_splitter as table_image_splitter

ocr_dir = cfg['fujitsu']['medical-receipt']['ocr-dir']
debug_flag = cfg['debug']
debug_dir = cfg["debug-dir"]
output_dir = cfg['fujitsu']['medical-receipt']['output-dir']


def convert_to_gray(img):
    dim = len(list(img.shape))
    if (dim == 3):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img.copy()
    return gray_image


def convert_to_binary(img):
    dim = len(list(img.shape))
    gray_image = convert_to_gray(img)
    bw = cv2.adaptiveThreshold(~gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    return bw


def get_image_intersection(img1, img2):
    return cv2.bitwise_and(img1, img2)


def get_blank_image(img):
    return np.zeros((img.shape[0], img.shape[1]), dtype=int)


def detect_horizontal(bw, scale=25, excess_size=5, debug_filename="file1"):
    horizontal = bw
    horizontal_size = len(horizontal[0]) / scale
    horizontal_structure_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (int(horizontal_size), 1))
    horizontal_structure_dilation = cv2.getStructuringElement(cv2.MORPH_RECT, (int(horizontal_size + excess_size), 1))
    horizontal = cv2.erode(horizontal, horizontal_structure_erosion, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontal_structure_dilation, (-1, -1))
    if debug_flag:
        cv2.imwrite(debug_dir + debug_filename + '_detectHor' + '.png', horizontal)
    return horizontal


def detect_dotted_horizontal_lines(bw, scale=25, debug_filename="file1"):
    horizontal = bw
    horizontal_size = len(horizontal[0]) / scale
    horizontal_structure_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (int(horizontal_size), 1))
    horizontal = cv2.erode(horizontal, horizontal_structure_erosion, (-1, -1))
    if debug_flag:
        cv2.imwrite(debug_dir + debug_filename + '_detectDotHor' + '.png', horizontal)
    return horizontal


def detect_vertical(bw, scale=25, excess_size=5, debug_filename="file1"):
    vertical = bw
    vertical_size = len(vertical[1]) / scale
    vertical_structure_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(vertical_size)))
    vertical_structure_dilation = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(vertical_size + excess_size)))
    vertical = cv2.erode(vertical, vertical_structure_erosion, (-1, -1))
    vertical = cv2.dilate(vertical, vertical_structure_dilation, (-1, -1))
    if debug_flag:
        cv2.imwrite(debug_dir + debug_filename + '_detectVert' + '.png', vertical)
    return vertical


def detect_dotted_vertical_lines(bw, scale=25, debug_filename="file1"):
    vertical = bw
    vertical_size = len(vertical[1]) / scale
    vertical_structure_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(vertical_size)))
    vertical = cv2.erode(vertical, vertical_structure_erosion, (-1, -1))
    if debug_flag:
        cv2.imwrite(debug_dir + debug_filename + '_detectDotVert' + '.png', vertical)
    return vertical


# ................................OLD DOTTED LINES START.............................................................
# def detect_dotted_lines(img, img_file, debug_filename="file1"):
#     file_name = '.'.join(img_file.split('.')[:-1])
#     gray_image = image.convert_to_gray(img)
#     bw = image.convert_to_binary(gray_image)
#     ocr_file = ocr_dir + file_name + ".hocr"
#     bw = ocr.clean_img_wrapper(bw, img_file, ocr_file, "hocr")
#     horizontal = detect_horizontal(bw, 40)
#     vertical = detect_vertical(bw, 40)
#     mask = horizontal + vertical
#     if debug_flag:
#         cv2.imwrite(debug_dir + debug_filename + "_wo_dotted.png", mask)
#     img_sobel = cv2.Sobel(gray_image, cv2.CV_8U, 1, 0, 3)
#     if debug_flag:
#         cv2.imwrite(debug_dir + debug_filename + "_sobel.png", img_sobel)
#     _, img_threshold = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
#     if debug_flag:
#         cv2.imwrite(debug_dir + debug_filename + "_sobelThresh.png", img_threshold)
#     element = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 8))
#     img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, element)
#     if debug_flag:
#         cv2.imwrite(debug_dir + debug_filename + "_sobelClose.png", img_threshold)
#     contours = cv2.findContours(img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     blank_img = mask * 0
#     cv2.drawContours(blank_img, contours[1], -1, (255, 255, 255), 3)
#     horizontal, vertical = blank_img, blank_img
#     horizontal = detect_dotted_horizontal_lines(horizontal, 40)
#     vertical = detect_dotted_vertical_lines(vertical, 40)
#     new_img = horizontal + vertical
#     mask += new_img
#     if (debug_flag):
#         cv2.imwrite(debug_dir + debug_filename + "_final.png", mask)
#     return mask

# ..........................OLD DOTTED LINES END..................................................................

# ...........................NEW DOTTED LINES START...............................................................

def detect_dotted_lines(img, img_file, debug_filename="file1"):
    file_name = '.'.join(img_file.split('.')[:-1])
    gray_image = convert_to_gray(img)
    bw = convert_to_binary(gray_image)
    # ocr_file = ocr_dir + file_name + ".hocr"
    # bw = ocr.clean_img_wrapper(bw, img_file, ocr_file, "hocr")
    # horizontal = detect_horizontal(bw, 40)
    # vertical = detect_vertical(bw, 40)
    # mask = horizontal + vertical

    # if debug_flag:
    #    cv2.imwrite(debug_dir + debug_filename + "_wo_dotted.png", mask)
    img_sobel = cv2.Sobel(gray_image, cv2.CV_8U, 1, 0, 3)
    if debug_flag:
        cv2.imwrite(debug_dir + debug_filename + "_sobel.png", img_sobel)

    _, img_threshold = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    if debug_flag:
        cv2.imwrite(debug_dir + debug_filename + "_sobelThresh.png", img_threshold)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 8))
    img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, element)
    if debug_flag:
        cv2.imwrite(debug_dir + debug_filename + "_sobelClose.png", img_threshold)

    contours = cv2.findContours(img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    blank_img = bw * 0
    cv2.drawContours(blank_img, contours[1], -1, (255, 255, 255), 3)
    if debug_flag:
        cv2.imwrite(debug_dir + debug_filename + "_contourDotted.png", blank_img)

    horizontal, vertical = blank_img, blank_img
    horizontal = detect_horizontal(horizontal, scale=60, excess_size=0)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 1))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, element)

    if debug_flag:
        cv2.imwrite(debug_dir + debug_filename + "_dottedHor.png", horizontal)

    vertical = detect_dotted_vertical_lines(vertical, 30)
    if debug_flag:
        cv2.imwrite(debug_dir + debug_filename + "_dottedVert.png", vertical)

    dotted_mask = horizontal + vertical
    if (debug_flag):
        cv2.imwrite(debug_dir + debug_filename + "_dotted_mask.png", dotted_mask)

    return dotted_mask


# ...........................NEW DOTTED LINES END.................................................................


def get_average_character_size(edges):
    import cv2
    previous_h = None
    previous_v = None
    previous_drop_h = None
    previous_drop_v = None
    final_h = None
    final_v = None
    for i in range(10, 100, 5):
        try:
            kernel_erosion_horizontal = np.ones((1, i), np.uint8)
            kernel_erosion_vertical = np.ones((i, 1), np.uint8)
            eroded_image_horizontal = cv2.morphologyEx(edges, cv2.MORPH_ERODE, kernel_erosion_horizontal)
            count_h = len(eroded_image_horizontal[eroded_image_horizontal > 200])
            eroded_image_vertical = cv2.morphologyEx(edges, cv2.MORPH_ERODE, kernel_erosion_vertical)
            count_v = len(eroded_image_vertical[eroded_image_vertical > 200])

            if previous_h is not None and previous_v is not None:
                drop_h = (previous_h - count_h) / previous_h
                drop_v = (previous_v - count_v) / previous_v
                if previous_drop_h is not None and drop_h > previous_drop_h and final_h is None:
                    final_h = i - 5

                if previous_drop_v is not None and drop_v > previous_drop_v and final_v is None:
                    final_v = i - 5

                previous_drop_h = drop_h
                previous_drop_v = drop_v

            previous_h = count_h
            previous_v = count_v
        except:
            pass

    if final_h is None:
        final_h = 50
    if final_v is None:
        final_v = 30

    return final_h, final_v


def detect_vertical_and_clean_text(bw, box_dim_cutoff, scale=25, excess_size=5, debug_filename="file1"):
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    vertical = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, morphKernel)
    verticalsize = len(vertical[1]) / scale
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(verticalsize)))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    _, contours, hierarchy = cv2.findContours(vertical, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    width, height = box_dim_cutoff
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if ((h <= height) and (w <= width)):
            th = 0
            cv2.rectangle(vertical, (x - th, y - th), (x + w - th, y + h - th), (0, 0, 0), -1)
        else:

            #### dual line fix due to char attached to line #############
            try:
                alllwed_width = 8
                if w > alllwed_width:
                    # print(w,x)
                    col_sum = vertical[y:y + h, x:x + w].sum(axis=0)
                    white_cols = np.where((np.array(col_sum) / (h * 255)) > 0.6)
                    if len(white_cols) > 0:
                        all_series = []
                        series = []
                        best_series = []
                        start = -1
                        for a in np.nditer(white_cols):
                            a = int(a)
                            if start == -1:
                                start = a
                                series.append(a)
                                continue
                            if abs(a - start) <= 3:
                                series.append(a)
                            else:
                                all_series.append(series)
                                if len(best_series) < len(series):
                                    best_series = series
                                series = []
                            start = a
                        all_series.append(series)
                        best_series = series
                        import math
                        if len(best_series) > 0:
                            min_series = best_series[0]
                            max_series = best_series[-1]
                            difference = abs(max_series - min_series)
                            if difference < alllwed_width:
                                half_dev = math.ceil((alllwed_width - difference) / 2)
                                if (max_series + half_dev) > w:
                                    min_series = min_series - half_dev - (max_series + half_dev - w)
                                    max_series = w
                                elif (min_series - half_dev) < 0:
                                    max_series = max_series + half_dev + abs(max_series - half_dev)
                                    min_series = 0
                                else:
                                    max_series = max_series + half_dev
                                    min_series = min_series - half_dev
                            w = max_series - min_series
                            if w > alllwed_width:
                                half_dev = math.ceil((w - alllwed_width) / 2)
                                w = alllwed_width
                                x = x + w - alllwed_width
                            else:
                                x = x + min_series
            except:
                pass
                # print(w,x)
            ################################
            th = 0
            cv2.rectangle(vertical, (x - th, y - th), (x + w + th, y + h + th), (255, 255, 255), -1)
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    # import random
    # cv2.imwrite(output_dir + debug_filename + str(random.randint)+'_newVertMask' + '.png', vertical)
    if debug_flag:
        cv2.imwrite(debug_dir + debug_filename + '_newVertMask' + '.png', vertical)
    return vertical

# def remove_text(rgb,rgb_line):
#     hasText = 0
#     rgb_cpy = rgb.copy()
#     rgb_line_cpy = rgb_line.copy()
#     gray = cv2.cvtColor(rgb_cpy, cv2.COLOR_BGR2GRAY);
#     morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#     grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, morphKernel)
#
#     # binarize
#     _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
#     # find contours
#     mask = np.zeros(bw.shape[:2], dtype="uint8")
#     _,contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#
#     # filter contours
#     idx = 0
#     widths = []
#     heights = []
#     while idx >= 0:
#         x,y,w,h = cv2.boundingRect(contours[idx])
#         widths.append(w)
#         heights.append(w)
#         idx = hierarchy[0][idx][0]
#
#     print('Mean Width', np.mean(widths))
#     print('Min Width', np.min(widths))
#     print('Max Width', np.max(widths))
#     print('Mean Height', np.mean(heights))
#     print('Min Height', np.min(heights))
#     print('Max Height', np.max(heights))
#
#     min_width = np.mean(widths) /2
#     max_width = np.mean(widths) + min_width
#     min_height = np.mean(widths) /2
#     max_height = np.mean(widths) + min_width
#
#     print('Min Width', min_width)
#     print('Max Width', max_width)
#     print('Min Height', max_height)
#     print('Max Height', max_height)
#     print(hierarchy)
#     idx = 0
#     while idx >= 0:
#         x,y,w,h = cv2.boundingRect(contours[idx]);
#         # ratio of non-zero pixels in the filled region
#         r = cv2.contourArea(contours[idx])/(w*h)
#         if((h < 7   or w < 7   ))  :
#             print((w,h,r))
#         else:
#             cv2.drawContours(rgb_line_cpy, contours, idx, (255, 255, 255), cv2.FILLED)
#         idx = hierarchy[0][idx][0]
#
#     return rgb_line_cpy

# def tabular_image_deskew(img, file_name):
#     # clean_image = image_cleaner.remove_shades(img, True, file_name)
#     img_cpy = img.copy()
#     table_rois, bboxline_coords = table_image_splitter.split_image_2_table_images(img_cpy, file_name)
#     # counter=0
#     max_angle = 0
#     for roi in table_rois:
#         roi = roi.astype(np.uint8)
#         # cv2.imwrite('image'+str(counter)+'.jpg',roi)
#         coords = np.column_stack(np.where(roi > 0))
#         angle = cv2.minAreaRect(coords)[-1]
#
#         if angle < -45:
#             angle = -(90 + angle)
#         else:
#             angle = -angle
#         if abs(angle) > abs(max_angle):
#             max_angle = angle
#         # (h, w) = roi.shape[:2]
#         # center = (w // 2, h // 2)
#         # M = cv2.getRotationMatrix2D(center, angle, 1.0)
#         # rotated = cv2.warpAffine(roi, M, (w, h),
#         #                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#
#         # cv2.imwrite('image_rotated' + str(counter) + '.jpg', rotated)
#         # counter+=1
#
#     (h, w) = img_cpy.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, max_angle, 1.0)
#     rotated = cv2.warpAffine(img_cpy, M, (w, h),
#                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     cv2.imwrite(output_dir + file_name + '_rotated.jpg', rotated)
#     return rotated