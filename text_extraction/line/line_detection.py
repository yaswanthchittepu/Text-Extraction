import numpy as np
import pandas as pd
import os
import apporchid.common.utils as utils
from operator import itemgetter
import copy
import cv2
import json
from apporchid.common.logger import logger as logger
from apporchid.common.config import cfg as cfg
#import apporchid.agora.cv.line.line_operations as lineop
import apporchid.agora.cv.line.line_properties as lineprop
from apporchid.agora.cv.line import line_operations as lineop
import random

PROJECT = cfg['project']
SUBPROJECT = cfg['subproject']

X_MAX = cfg[PROJECT][SUBPROJECT]["x-max"]
Y_MAX = cfg[PROJECT][SUBPROJECT]["y-max"]
MIN_LINE_LENGTH = cfg[PROJECT][SUBPROJECT]["hough-transform"]["min-line-length"]
MAX_LINE_GAP = cfg[PROJECT][SUBPROJECT]["hough-transform"]["max-line-gap"]
RHO_VALUE = cfg[PROJECT][SUBPROJECT]["hough-transform"]["rho-value"]
LINE_LENGTH_THRESHOLD = cfg[PROJECT][SUBPROJECT]["line-detection"]["line-length-threshold"]
X_DEVIATION = cfg[PROJECT][SUBPROJECT]["horizontal-line-join"]["x-deviation"]
Y_DEVIATION = cfg[PROJECT][SUBPROJECT]["horizontal-line-join"]["y-deviation"]
Y_DEVIATION_VERTCAL_DISTANCE = cfg[PROJECT][SUBPROJECT]["vertical-line-join"][
    "y-deviation-vertical-distance"]
BORDER_PIXELS_TO_REMOVE = cfg[PROJECT][SUBPROJECT]["line-detection"]["border-pixels-to-remove"]
LINE_POINT_X_DISTANCE = cfg[PROJECT][SUBPROJECT]["line-point-x-distance"]
LINE_POINT_Y_DISTANCE = cfg[PROJECT][SUBPROJECT]["line-point-y-distance"]
debug_dir = cfg["debug-dir"]
debug_flag = cfg['debug']

def detect_lines_in_images(images):
    
    for img_clean in images:
        for roi in img_clean:
            roi = roi.astype(np.uint8)
            lines = detect_lines(roi)


def detect_lines(img_clean, bbox_lines, orig_image):

    try:
        # ''' A: find lines by hough transformation'''
        lines = cv2.HoughLinesP(img_clean, RHO_VALUE, np.pi / 180, 10, MIN_LINE_LENGTH, MAX_LINE_GAP)
        lines = utils.ndarray_to_list(lines)

        lines += bbox_lines
        if debug_flag:
            write_lines_to_image(orig_image, lines, suffix='houghTransform')

        # Invert colors for Contour transform
        img_clean = 255 - img_clean

        _, contours_clean, _ = cv2.findContours(img_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        lines_countours = find_vertical_lines_from_countour(contours_clean)
        lines = lines + lines_countours

        ''' D: find lines on clean images by Hough transformation'''
        # lines_clean = cv2.HoughLinesP(~img_clean,rhoValue,np.pi/180,10,minLineLength,maxLineGap)
        # lines_clean = ndarray_to_list(lines_clean)
        # lines = lines + lines_clean

        ''' Selecting unique list of lines'''
        lines = [list(y) for y in set(tuple(x) for x in lines)]

        ''' Custom transformations'''
        lines = lineop.arrange_lines_AB_order(lines)
        if debug_flag:
            write_lines_to_image(orig_image, lines, suffix='arrangeLines')
        lines = lineop.remove_short_lines(lines, LINE_LENGTH_THRESHOLD - 20)
        if debug_flag:
            write_lines_to_image(orig_image, lines, suffix='removeShortLines')
        lines = lineop.join_horizontal_lines(lines)
        if debug_flag:
            write_lines_to_image(orig_image, lines, suffix='joinHorLines')
        lines = lineop.join_vertical_lines(lines)
        if debug_flag:
            write_lines_to_image(orig_image, lines, suffix='joinVertLines')
        lines = lineop.remove_short_lines(lines, LINE_LENGTH_THRESHOLD + 20)
        if debug_flag:
            write_lines_to_image(orig_image, lines, suffix='removeShortLines')
        lines = lineop.remove_left_right_borders(lines)
        if debug_flag:
            write_lines_to_image(orig_image, lines, suffix='removeBorders')
        # lines = lineop.extend_horizontal_lines(lines)
        lines = lineop.merge_lines(lines, lines)
        if debug_flag:
            write_lines_to_image(orig_image, lines, suffix='mergeLines')
        return lines
    except Exception as e:
        logger.warn(e)
        raise e


def find_vertical_lines_from_countour(contours):
    
    previous = None
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for contour in contours:
        for i in range(len(contour)):
            if i > 0:
                x1.append(contour[i][0][0])
                y1.append(contour[i][0][1])
                x2.append(previous[0])
                y2.append(previous[1])
            previous = [contour[i][0][0], contour[i][0][1]]
    condf = pd.DataFrame({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    condf.reset_index(drop=True)
    data = condf.copy()
    data['xdiff'] = abs(data['x1'] - data['x2'])
    data['ydiff'] = abs(data['y1'] - data['y2'])

    vertical = data[(data['x1'] == data['x2']) & (data.ydiff > 5)]
    vertical = vertical.drop(['xdiff'], axis=1)
    vertical = vertical.sort_values('ydiff')

    vertical.reset_index(drop=True)
    # filter small lines
    verrtical_biglines = vertical[vertical.ydiff > 10]

    # find horizontal lines
    horizontal = data[(data['y1'] == data['y2']) & (data.xdiff > 5)]
    horizontal = horizontal.drop(['ydiff'], axis=1)
    horizontal = horizontal.sort_values('xdiff')
    # horizontal.index = range(len(horizontal))
    horizontal.reset_index(drop=True)
    # filter small lines
    horizontal_biglines = horizontal[horizontal.xdiff > 50]

    lines = []
    for _, row in verrtical_biglines.iterrows():
        lines.append([row['x1'], row['y1'], row['x2'], row['y2']])

    for _, row in horizontal_biglines.iterrows():
        lines.append([row['x1'], row['y1'], row['x2'], row['y2']])

    return lines

# .......................................OLD MERGE LINES END....................................................





def get_bounding_boxes_from_ocr(json_char_filename):
    
    with open(json_char_filename, 'r', encoding="utf8") as f:
        data = json.load(f)
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for pages in data['pages']:
        for sections in pages['sections']:
            for sectionEntitiesList in sections['sectionEntitiesList']:
                for lines in sectionEntitiesList['lines']:
                    for words in lines['words']:
                        for char in words['characters']:
                            if (char['bbox']['x2'] - char['bbox']['x1'] < 500 
                                and char['bbox']['y2'] - char['bbox']['y1'] < 100 
                                and char['value'][0] != '-' 
                                and char['value'][0] != '|' and char['value'][0] != '.'):
                                
                                x1.append(char['bbox']['x1'])
                                y1.append(char['bbox']['y1'])
                                x2.append(char['bbox']['x2'])
                                y2.append(char['bbox']['y2'])

            for sectionEntities in sections['sectionEntities']:
                for lines in sectionEntities['lines']:
                    for words in lines['words']:
                        for char in words['characters']:
                            if char['bbox']['x2'] - char['bbox']['x1'] < 500 and char['bbox']['y2'] - char['bbox'][
                                'y1'] < 100 \
                                    and char['value'][0] != '-' and char['value'][0] != '|' and char['value'][
                                0] != '.':
                                x1.append(char['bbox']['x1'])
                                y1.append(char['bbox']['y1'])
                                x2.append(char['bbox']['x2'])
                                y2.append(char['bbox']['y2'])
    boxes = pd.DataFrame({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    return boxes


def split_vertical_lines(support_lines, long_lines, boxes):
    
    # v3 = longlines, v1 = supportlines

    newlines = pd.DataFrame([], columns=support_lines.columns)
    ignore = long_lines.columns.get_loc('ignore')
    long_lines.index = range(len(long_lines))

    for index, row in long_lines.iterrows():
        for _, row1 in boxes.iterrows():
            result = lineprop.intersection([row.x1, row.y1, row.x2, row.y2], [row1.x1, row1.y1, row1.x2, row1.y1])
            if result[0]:
                result = lineprop.intersection([row.x1, row.y1, row.x2, row.y2], [row1.x1, row1.y2, row1.x2, row1.y2])
                if result[0]:
                    # see if there is similar vertical line from run 1
                    if (len(support_lines[(support_lines.x1 > row.x1 - 10) & (support_lines.x1 < row.x1 + 10) & (
                            support_lines.y1 < row1.y1) & (support_lines.y2 > row1.y2)])):
                        continue
                    # break the vertical line in two
                    # chcek the order of columns in long_lines
                    newlines = pd.concat([newlines, pd.DataFrame([[row.x1, row.y1, row.x2, int(row1.y1), 0, 0]],
                                                                 columns=long_lines.columns,
                                                                 index=[len(long_lines)])])
                    newlines = pd.concat([newlines, pd.DataFrame([[row.x1, int(row1.y2), row.x2, row.y2, 0, 0]],
                                                                 columns=long_lines.columns,
                                                                 index=[len(long_lines)])])
                    # support_lines = pd.concat([support_lines,df])
                    long_lines.iloc[index, ignore] = 1
                    break  # need to check
    new_long_lines = long_lines[long_lines.ignore == 0]

    return new_long_lines, newlines


def process_vertical_lines_trunc_ocr(support_lines, long_lines, boxes):
    
    # same as above

    # for vertical lines, split the files for the first time
    non_ignored_lines, newlines = split_vertical_lines(support_lines, long_lines, boxes)
    new_vertical_lines = pd.concat([non_ignored_lines, newlines], ignore_index=True)

    # Iterate and see if more splits are possible
    if len(new_vertical_lines) > len(long_lines):
        new_lines_from_iteration = pd.DataFrame([], columns=long_lines.columns)
        # logger.debug('Starting second pass ')
        while (len(newlines) != len(new_lines_from_iteration)):
            if len(new_lines_from_iteration) > 0:
                newlines = new_lines_from_iteration

            non_ignored_lines, newlines_1 = split_vertical_lines(support_lines, newlines, boxes)
            new_lines_from_iteration = pd.concat([non_ignored_lines, newlines_1], ignore_index=True)

        new_vertical_lines = pd.concat([new_vertical_lines, new_lines_from_iteration], ignore_index=True)

    return new_vertical_lines


def truncate_lines_from_ocr(json_char_filename, support_lines, final_lines):
    
    # l3 = finallines l1 = supportlines

    boxes = get_bounding_boxes_from_ocr(json_char_filename)

    lines_from_run1 = pd.DataFrame(support_lines, columns=['x1', 'y1', 'x2', 'y2'])
    final_lines = pd.DataFrame(final_lines, columns=['x1', 'y1', 'x2', 'y2'])

    horizontal_lines_support, vertical_lines_support = lineop.preprocess_df(lines_from_run1)
    horizontal_lines_final, vertical_lines_final = lineop.preprocess_df(final_lines)
    vertical_lines_final['ignore'] = 0

    new_vertical_lines = process_vertical_lines_trunc_ocr(vertical_lines_support, vertical_lines_final, boxes)

    finaldata = pd.concat([new_vertical_lines, horizontal_lines_final], ignore_index=True)
    finaldata = finaldata[['x1', 'y1', 'x2', 'y2']]
    finaldata = finaldata.values.tolist()

    return finaldata
    '''
    fname = '.'.join(img_file.split('.')[:-1])
    output_image = lines_output_dir + fname + '_' + suffix + '.png'

    img_cpy = copy.deepcopy(image)
    for line in lines:
        x1, y1, x2, y2 = line
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.line(img_cpy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(output_image, img_cpy)
    '''

def write_lines_to_image(image, lines, suffix):
    output_image = debug_dir + suffix + str(random.randint(1,26)) + '.png'
    img_cpy = copy.deepcopy(image)
    for line in lines:
        x1, y1, x2, y2 = line
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.line(img_cpy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(output_image, img_cpy)
#....................................................NEW MERGE LINES START.......................................

# def process_vertical_lines(vertical_df, vertical_df_evidence):
#     ################
#     # Processing Vertical lines
#     ################
#
#     def _get_nearby_lines(vertical_df):
#
#         NEARBY_RANGE = 15
#         nearby_lines = vertical_df[(vertical_df.x1 > main_row.x1 - NEARBY_RANGE) & \
#                                    (vertical_df.x1 < main_row.x1 + NEARBY_RANGE) & \
#                                    (vertical_df.ignore == 0)]
#
#         return nearby_lines
#
#     def _is_extendable(nearby_lines):
#
#         nearby_lines['index'] = nearby_lines.index.tolist()
#         nearby_lines.reset_index(inplace=True, drop=True)
#
#         dont_extend = False
#         for _, support_row in nearby_lines.iterrows():
#
#             # check if this is part of some line in temp
#
#             # If both row and row1 are same, index will match
#             if (main_row_idx == support_row['index']): continue
#
#             # If some line in support row is bigger than main row, ignore main_row
#             if ((support_row.y1 <= main_row.y1) & (support_row.y2 >= main_row.y2)):
#                 dont_extend = True
#                 vertical_df.iloc[main_row_idx, ignore] = 1
#                 break
#
#         return dont_extend
#
#     def _bool_support_present(vertical_df_evidence, main_row, previous, rowy2):
#         if len(vertical_df_evidence[(vertical_df_evidence.x1 > main_row.x1 - 5) & \
#                                     (vertical_df_evidence.x1 < main_row.x1 + 5) & \
#                                     (vertical_df_evidence.y1 <= previous + 3) & \
#                                     (vertical_df_evidence.y2 >= rowy2 - 3)]
#                ) > 0:
#             return True
#
#         else:
#             return False
#
#     vertical_df['ignore'] = 0
#     ignore = vertical_df.columns.get_loc('ignore')
#     y2loc = vertical_df.columns.get_loc('y2')
#
#     vertical_df = vertical_df.sort_values('y1')
#
#     for main_row_idx, main_row in vertical_df.iterrows():
#
#         if main_row.ignore == 1:
#             continue
#
#         nearby_lines = _get_nearby_lines(vertical_df)
#
#         dont_extend = _is_extendable(nearby_lines)
#
#         if dont_extend: continue
#
#         # extend iteratively by 5 pixels and exit when no further evidence present to extend
#         previous = main_row.y2
#         rowy2 = main_row.y2
#
#         while (True):
#             rowy2 += 10
#             # check for the evidence in horizontal_df_evidence
#             if _bool_support_present(vertical_df_evidence, main_row, previous, rowy2):
#                 previous = rowy2
#
#             else:  # No further support available
#                 rowy2 -= 10
#                 break
#
#         vertical_df.iloc[main_row_idx, y2loc] = rowy2
#
#     return vertical_df
#
#
# def process_horizontal_lines(horizontal_df, horizontal_df_evidence):
#     ################
#     # Processing Horizontal lines
#     ################
#
#     def _get_nearby_lines(horizontal_df):
#
#         NEARBY_RANGE = 15
#         nearby_lines = horizontal_df[(horizontal_df.y1 > main_row.y1 - NEARBY_RANGE) & \
#                                      (horizontal_df.y1 < main_row.y1 + NEARBY_RANGE) & \
#                                      (horizontal_df.ignore == 0)]
#
#         return nearby_lines
#
#     def _is_extendable(nearby_lines):
#
#         nearby_lines['index'] = nearby_lines.index.tolist()
#         nearby_lines.reset_index(inplace=True, drop=True)
#
#         dont_extend = False
#         for _, support_row in nearby_lines.iterrows():
#
#             # check if this is part of some line in temp
#
#             # If both row and row1 are same, index will match
#             if (main_row_idx == support_row['index']): continue
#
#             # If some line in support row is bigger than main row, ignore main_row
#             if ((support_row.x1 <= main_row.x1 + 5) & (support_row.x2 >= main_row.x2 - 5)):
#                 dont_extend = True
#                 horizontal_df.iloc[main_row_idx, ignore] = 1
#                 break
#
#         return dont_extend
#
#     def _bool_support_present(horizontal_df_evidence, main_row, previous, rowx2):
#         if len(horizontal_df_evidence[(horizontal_df_evidence.y1 > main_row.y1 - 2) & \
#                                       (horizontal_df_evidence.y1 < main_row.y1 + 2) & \
#                                       (horizontal_df_evidence.x1 <= previous) & \
#                                       (horizontal_df_evidence.x2 >= rowx2)]
#                ) > 0:
#             return True
#
#         else:
#             return False
#
#     horizontal_df['ignore'] = 0
#     ignore = horizontal_df.columns.get_loc('ignore')
#     x2loc = horizontal_df.columns.get_loc('x2')
#
#     horizontal_df = horizontal_df.sort_values('x1')
#
#     for main_row_idx, main_row in horizontal_df.iterrows():
#
#         if main_row.ignore == 1:
#             continue
#
#         nearby_lines = _get_nearby_lines(horizontal_df)
#
#         dont_extend = _is_extendable(nearby_lines)
#
#         if dont_extend: continue
#
#         # extend iteratively by 5 pixels and exit when no further evidence present to extend
#         previous = main_row.x2
#         rowx2 = main_row.x2
#
#         while (True):
#             rowx2 += 5
#             # check for the evidence in horizontal_df_evidence
#             if _bool_support_present(horizontal_df_evidence, main_row, previous, rowx2):
#                 previous = rowx2
#
#             else:  # No further support available
#                 rowx2 -= 5
#                 break
#
#         horizontal_df.iloc[main_row_idx, x2loc] = rowx2
#
#     return horizontal_df
#
#
# def make_lines_horizontal(lines):
#     lines['y2'].loc[lines['xdiff'] > lines['ydiff']] = lines['y1']
#     return lines
#
#
# def make_lines_vertical(lines):
#     lines['x2'].loc[lines['xdiff'] < lines['ydiff']] = lines['x1']
#     return lines
#
#
# def separate_horizontal_vertical_lines(lines):
#     horizontal_lines = lines[lines['y1'] == lines['y2']]
#     horizontal_lines = horizontal_lines.drop(['ydiff'], axis=1)
#     horizontal_lines = horizontal_lines.sort_values('xdiff')
#     horizontal_lines.index = range(len(horizontal_lines))
#
#     vertical_lines = lines[lines['x1'] == lines['x2']]
#     vertical_lines = vertical_lines.drop(['xdiff'], axis=1)
#     vertical_lines = vertical_lines.sort_values('ydiff')
#     vertical_lines.index = range(len(vertical_lines))
#
#     return horizontal_lines, vertical_lines
#
#
# def fix_corordinates_ordering(horizontal_lines, vertical_lines):
#     def _fix_horizontal_lines(horizontal_lines):
#         horizontal_lines['temp_x2'] = horizontal_lines['x2']
#         horizontal_lines['x2'].loc[horizontal_lines['x2'] < horizontal_lines['x1']] = horizontal_lines['x1']
#         horizontal_lines['x1'].loc[horizontal_lines['temp_x2'] < horizontal_lines['x1']] = horizontal_lines['temp_x2']
#         horizontal_lines.drop(['temp_x2'], axis=1, inplace=True)
#         return horizontal_lines
#
#     def _fix_vertical_lines(vertical_lines):
#         vertical_lines['temp_y2'] = vertical_lines['y2']
#         vertical_lines['y2'].loc[vertical_lines['y2'] < vertical_lines['y1']] = vertical_lines['y1']
#         vertical_lines['y1'].loc[vertical_lines['temp_y2'] < vertical_lines['y1']] = vertical_lines['temp_y2']
#         vertical_lines.drop(['temp_y2'], axis=1, inplace=True)
#         return vertical_lines
#
#     horizontal_lines = _fix_horizontal_lines(horizontal_lines)
#     vertical_lines = _fix_vertical_lines(vertical_lines)
#
#     return horizontal_lines, vertical_lines
#
#
# def preprocess_lines(lines):
#     lines['xdiff'] = abs(lines['x1'] - lines['x2'])
#     lines['ydiff'] = abs(lines['y1'] - lines['y2'])
#
#     lines = make_lines_horizontal(lines)
#     lines = make_lines_vertical(lines)
#
#     # split the data into vertical_lines and horizontal_lines lines
#     horizontal_lines, vertical_lines = separate_horizontal_vertical_lines(lines)
#     horizontal_lines, vertical_lines = fix_corordinates_ordering(horizontal_lines, vertical_lines)
#
#     return horizontal_lines, vertical_lines
#
#
# def lines_initializer(line1, line3):
#     # Create DataFrames
#     support_lines_data = pd.DataFrame(line1, columns=['x1', 'y1', 'x2', 'y2'])
#     final_lines_data = pd.DataFrame(line3, columns=['x1', 'y1', 'x2', 'y2'])
#
#     # preprcess dataframes to arrange coordinates and make perfect vertical / horizontal
#     horizontal_lines_support, vertical_lines_support = preprocess_lines(support_lines_data)
#     horizontal_lines_final, vertical_lines_final = preprocess_lines(final_lines_data)
#
#     return horizontal_lines_support, vertical_lines_support, horizontal_lines_final, vertical_lines_final
#
#
# def merge_lines(support_lines, final_lines):
#     (horizontal_lines_support, vertical_lines_support,
#      horizontal_lines_final, vertical_lines_final) = lines_initializer(support_lines, final_lines)
#
#     horizontal = process_horizontal_lines(horizontal_lines_support, horizontal_lines_final)
#     vertical = process_vertical_lines(vertical_lines_support, vertical_lines_final)
#
#     # update temp as well as horizontal1
#     #################
#     ##Desired output#
#     #################
#
#     horizontal = horizontal[horizontal.ignore == 0]
#     horizontal = horizontal[['x1', 'y1', 'x2', 'y2']]
#     vertical = vertical[vertical.ignore == 0]
#     vertical = vertical[['x1', 'y1', 'x2', 'y2']]
#
#     lines = pd.concat([vertical, horizontal], ignore_index=True)
#     lines = lines.values.tolist()
#
#     return lines

#....................................................NEW MERGE LINES END..........................................