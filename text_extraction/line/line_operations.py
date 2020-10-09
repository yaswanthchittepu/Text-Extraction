import pandas as pd
from operator import itemgetter
import numpy as np
import copy
from apporchid.common.logger import logger
from apporchid.common.config import cfg as cfg
import apporchid.agora.cv.line.line_properties as lineprop

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


def _merge_nearby_horizontal_lines(lines):
    Y_LAG = 15
    vertical_lines = []
    horizontal_lines = []
    for line in lines:
        if lineprop.slope(line) == float('inf'):
            vertical_lines.append(line)
        else:
            horizontal_lines.append(line)

    horizontal_df = pd.DataFrame(horizontal_lines, columns=['x1', 'y1', 'x2', 'y2'])
    vertical_df = pd.DataFrame(vertical_lines, columns=['x1', 'y1', 'x2', 'y2'])

    horizontal_df['xdiff'] = abs(horizontal_df['x1'] - horizontal_df['x2'])
    horizontal_df['ydiff'] = abs(horizontal_df['y1'] - horizontal_df['y2'])

    vertical_df['xdiff'] = abs(horizontal_df['x1'] - horizontal_df['x2'])
    vertical_df['ydiff'] = abs(horizontal_df['y1'] - horizontal_df['y2'])

    horizontal_df['ignore'] = 0
    vertical_df['ignore'] = 0

    horizontal_df['lag1'] = horizontal_df['y1'] - horizontal_df['y1'].shift(1)
    horizontal_df = horizontal_df.fillna(0.0)
    i = 1
    yccord = horizontal_df.columns.get_loc('lag1')
    for index, row in horizontal_df.iterrows():
        if row['lag1'] < Y_LAG:
            horizontal_df.iloc[index, yccord] = str(i) + 'a'
        else:
            i += 1  # save this i for further processing

    for j in range(i):

        temp1 = horizontal_df[horizontal_df['lag1'] == str(j + 1) + 'a']
        if temp1.empty:
            continue
        if j > 0:
            temp1 = pd.concat([pd.DataFrame(horizontal_df.iloc[temp1.index[0] - 1]).transpose(), temp1])
            for i in temp1.index:
                horizontal_df.iloc[i, horizontal_df.columns.get_loc('ignore')] = 1
            yvalue = temp1[temp1['xdiff'] == temp1.xdiff.max()].y1
            yvalue = yvalue.iloc(yvalue.index)[0]
            df = pd.DataFrame(
                [[temp1.x1.min(), yvalue, temp1.x2.max(), yvalue, abs(temp1.x1.min() - temp1.x2.max()), 0, 0, 0]],
                columns=horizontal_df.columns, index=[len(horizontal_df)])
            horizontal_df = pd.concat([horizontal_df, df])
        else:
            for i in temp1.index:
                horizontal_df.iloc[i, horizontal_df.columns.get_loc('ignore')] = 1
            yvalue = temp1[temp1['xdiff'] == temp1.xdiff.max()].y1
            yvalue = yvalue.iloc(yvalue.index)[0]
            df = pd.DataFrame(
                [[temp1.x1.min(), yvalue, temp1.x2.max(), yvalue, abs(temp1.x1.min() - temp1.x2.max()), 0, 0, 0]],
                columns=horizontal_df.columns, index=[len(horizontal_df)])
            horizontal_df = pd.concat([horizontal_df, df])

    complete_list = pd.concat([horizontal_df, vertical_df], ignore_index=True)
    complete_list = complete_list[complete_list['ignore'] == 0]
    complete_list = complete_list[['x1', 'y1', 'x2', 'y2']]
    lines = complete_list.values.tolist()
    return lines


def extend_horizontal_lines(lines):
    vertical_lines = []
    horizontal_lines = []
    for line in lines:
        if lineprop.slope(line) > 100:
            vertical_lines.append(line)
        elif lineprop.slope(line) == 0:
            horizontal_lines.append(line)

    # Add logic to don't extend if already a vertical line present in X axis

    horizontal_df = pd.DataFrame(horizontal_lines, columns=['x1', 'y1', 'x2', 'y2'])
    vertical_df = pd.DataFrame(vertical_lines, columns=['x1', 'y1', 'x2', 'y2'])

    horizontal_df['max_x'] = horizontal_df[['x1', 'x2']].max(axis=1)
    horizontal_df['min_x'] = horizontal_df[['x1', 'x2']].min(axis=1)
    horizontal_df['max_y'] = horizontal_df[['y1', 'y2']].max(axis=1)
    horizontal_df['min_y'] = horizontal_df[['y1', 'y2']].min(axis=1)

    vertical_df['max_x'] = vertical_df[['x1', 'x2']].max(axis=1)
    vertical_df['min_x'] = vertical_df[['x1', 'x2']].min(axis=1)
    vertical_df['max_y'] = vertical_df[['y1', 'y2']].max(axis=1)
    vertical_df['min_y'] = vertical_df[['y1', 'y2']].min(axis=1)

    x1coord = horizontal_df.columns.get_loc('x1')
    x2coord = horizontal_df.columns.get_loc('x2')
    for index, row in horizontal_df.iterrows():
        # extend left
        # if horizontal Y line falls in between Ymin and Ymax
        # Or, If it lies in some distance
        # And, It is left to it
        # And, left
        temp = vertical_df[(((vertical_df.max_y >= row.min_y) & (vertical_df.min_y <= row.min_y)) | (
                abs(vertical_df.min_y - row.min_y) < 30) | (abs(vertical_df.max_y - row.max_y) < 30)) & (
                                   vertical_df.max_x < row.min_x) & (abs(row.min_x - vertical_df.max_x) < 100)]  #
        if len(temp) > 0:
            # horizontal_df.iloc[index,x2coord] = horizontal_df.iloc[index].max_x
            horizontal_df.iloc[index, x1coord] = temp.min_x.max()
        # extend right

        temp = vertical_df[(((vertical_df.max_y >= row.min_y) & (vertical_df.min_y <= row.min_y)) | (
                abs(vertical_df.min_y - row.min_y) < 30) | (abs(vertical_df.max_y - row.max_y) < 30)) & (
                                   vertical_df.min_x > row.max_x) & (abs(vertical_df.min_x - row.max_x) < 100)]  #

        if len(temp) > 0:
            # horizontal_df.iloc[index,x1coord] = horizontal_df.iloc[index].min_x
            horizontal_df.iloc[index, x2coord] = temp.max_x.min()

    complete_list = pd.concat([horizontal_df, vertical_df], ignore_index=True)
    complete_list = complete_list[['x1', 'y1', 'x2', 'y2']]
    lines = complete_list.values.tolist()
    return lines





def arrange_lines(lines, first_element, second_element):
    ouput_lines = sorted(lines, key=itemgetter(first_element, second_element))
    return ouput_lines





def remove_short_lines(lines, length):
    # remove lines which has some min length and gradient of less than 1(45 degrees)
    # so that we dont remove short vertical lines
    # slope(line)<1 and
    long_lines = []
    for line in lines:
        if lineprop.line_length(line) < length:
            pass
        else:
            long_lines.append(line)
    return long_lines


def remove_short_vertical_lines(lines, length):
    # remove lines which has some min length and gradient of less than 1(45 degrees)
    # so that we dont remove short vertical lines
    # slope(line)<1 and
    long_lines = []
    for line in lines:
        if lineprop.line_length(line) < length and lineprop.slope(line) > 100:
            pass
        else:
            long_lines.append(line)
    return long_lines


def remove_short_horizontal_lines(lines, length):
    # remove lines which has some min length and gradient of less than 1(45 degrees)
    # so that we dont remove short vertical lines
    # slope(line)<1 and
    long_lines = []
    for line in lines:
        if lineprop.line_length(line) < length and lineprop.slope(line) < 1:
            pass
        else:
            long_lines.append(line)
    return long_lines


def arrange_lines_AB_order(lines):
    x1, y1, x2, y2 = 0, 1, 2, 3
    new_lines = []
    for line in lines:
        new_line = [None] * 4
        if line[x1] > line[x2]:
            # temp_coordinates=line[x1]
            # line[x1] = line[x2]
            # line[x2] = temp_coordinates
            new_line[x1] = line[x2]
            new_line[x2] = line[x1]
        else:
            new_line[x1] = line[x1]
            new_line[x2] = line[x2]

        if line[y1] > line[y2]:
            # temp_coordinates=line[y1]
            # line[y1] = line[y2]
            # line[y2] = temp_coordinates
            new_line[y1] = line[y2]
            new_line[y2] = line[y1]

        else:
            new_line[y1] = line[y1]
            new_line[y2] = line[y2]

        new_lines.append(new_line)

    return new_lines


def find_points_of_intersection_and_extend_lines(lines):
    points = []
    x1, y1, x2, y2 = 0, 1, 2, 3
    intersection_lines = []
    for line1 in lines:
        line1_copy = line1.copy()
        for line2 in lines:
            line2_copy = line2.copy()
            if line1 != line2 and abs(lineprop.slope(line1) - lineprop.slope(line2)) > 1:
                A = [line1[x1], line1[y1]]
                B = [line1[x2], line1[y2]]
                C = [line2[x1], line2[y1]]
                D = [line2[x2], line2[y2]]

                try:
                    X, Y = lineprop.line_intersection((A, B), (C, D))
                except:
                    continue
                X = int(X)
                Y = int(Y)
                if X < X_MAX and Y < Y_MAX and X > 0 and Y > 0:
                    # find slope of both lines
                    slope_line1 = lineprop.slope(line1)
                    slope_line2 = lineprop.slope(line2)

                    if slope_line1 < 1 and slope_line2 > 1:  # line1 is horizontal and line2 is vertical
                        if X <= min(line1[x1], line1[x2]) and Y <= min(line2[y1], line2[y2]):

                            # condition: line1 is right, line2 is below
                            if abs(X - min(line1[x1], line1[x2])) < LINE_POINT_X_DISTANCE and abs(
                                    Y - min(line2[y1], line2[y2])) < LINE_POINT_Y_DISTANCE:
                                points.append((X, Y))
                                if line1[x1] > X:
                                    line1_copy[x1] = X
                                if line2[y1] > Y:
                                    line2_copy[y1] = Y
                                intersection_lines.append(line1_copy)
                                intersection_lines.append(line2_copy)

                        # condition: line1 is left, line2 is  below
                        elif X >= max(line1[x1], line1[x2]) and Y <= min(line2[y1], line2[y2]):
                            if ((abs(X - max(line1[x1], line1[x2])) < LINE_POINT_X_DISTANCE and abs(
                                    Y - min(line2[y1], line2[y2])) < LINE_POINT_Y_DISTANCE)):
                                points.append((X, Y))
                                if line1[x2] < X:
                                    line1_copy[x2] = X
                                if line2[y1] > Y:
                                    line2_copy[y1] = Y
                                intersection_lines.append(line1_copy)
                                intersection_lines.append(line2_copy)

                        # condition: line1 is left, line2 is above
                        elif X >= max(line1[x1], line1[x2]) and Y >= max(line2[y1], line2[y2]):
                            if ((abs(X - max(line1[x1], line1[x2])) < LINE_POINT_X_DISTANCE and abs(
                                    Y - max(line2[y1], line2[y2])) < LINE_POINT_Y_DISTANCE)):
                                points.append((X, Y))
                                if line1[x2] < X:
                                    line1_copy[x2] = X
                                if line2[y2] < Y:
                                    line2_copy[y2] = Y
                                intersection_lines.append(line1_copy)
                                intersection_lines.append(line2_copy)

                        # condition: line1 is right, line2 is above
                        if X <= min(line1[x1], line1[x2]) and Y >= max(line2[y1], line2[y2]):
                            if ((abs(X - min(line1[x1], line1[x2])) < LINE_POINT_X_DISTANCE and abs(
                                    Y - max(line2[y1], line2[y2])) < LINE_POINT_Y_DISTANCE)):
                                points.append((X, Y))
                                if line1[x1] > X:
                                    line1_copy[x1] = X
                                if line2[y2] < Y:
                                    line2_copy[y2] = Y
                                intersection_lines.append(line1_copy)
                                intersection_lines.append(line2_copy)
                    elif slope_line1 > 1 and slope_line2 < 1:  # line1 is vertical and line2 is horizontal
                        # condition: line1 is right, line2 is below
                        if X <= min(line2[x1], line2[x2]) and Y <= min(line1[y1], line1[y2]):
                            if ((abs(X - min(line2[x1], line2[x2])) < LINE_POINT_X_DISTANCE and abs(
                                    Y - min(line1[y1], line1[y2])) < LINE_POINT_Y_DISTANCE)):
                                points.append((X, Y))
                                if line2[x1] > X:
                                    line2_copy[x1] = X
                                if line1[y1] > Y:
                                    line1_copy[y1] = Y
                                intersection_lines.append(line1_copy)
                                intersection_lines.append(line2_copy)

                        # condition: line1 is left, line2 is below
                        elif X >= max(line2[x1], line2[x2]) and Y <= min(line1[y1], line1[y2]):
                            if ((abs(X - max(line2[x1], line2[x2])) < LINE_POINT_X_DISTANCE and abs(
                                    Y - min(line1[y1], line1[y2])) < LINE_POINT_Y_DISTANCE)):
                                points.append((X, Y))
                                if line2[x2] < X:
                                    line2_copy[x2] = X
                                if line1[y1] > Y:
                                    line1_copy[y1] = Y
                                intersection_lines.append(line1_copy)
                                intersection_lines.append(line2_copy)

                        # condition: line1 is left, line2 is above
                        elif X >= max(line2[x1], line2[x2]) and Y >= max(line1[y1], line1[y2]):
                            if ((abs(X - max(line2[x1], line2[x2])) < LINE_POINT_X_DISTANCE and abs(
                                    Y - max(line1[y1], line1[y2])) < LINE_POINT_Y_DISTANCE)):
                                points.append((X, Y))
                                if line2[x2] < X:
                                    line2_copy[x2] = X
                                if line1[y2] < Y:
                                    line1_copy[y2] = Y
                                intersection_lines.append(line1_copy)
                                intersection_lines.append(line2_copy)

                        # condition: line1 is right, line2 is above
                        if X <= min(line2[x1], line2[x2]) and Y >= max(line1[y1], line1[y2]):
                            if ((abs(X - min(line2[x1], line2[x2])) < LINE_POINT_X_DISTANCE and abs(
                                    Y - max(line1[y1], line1[y2])) < LINE_POINT_Y_DISTANCE)):
                                points.append((X, Y))
                                if line2[x1] > X:
                                    line2_copy[x1] = X
                                if line1[y2] < Y:
                                    line1_copy[y2] = Y
                                intersection_lines.append(line1_copy)
                                intersection_lines.append(line2_copy)

                    # find is lines passes through these points
                    if slope_line1 < 1 and slope_line2 > 1:  # line1 is horizontal and line2 is vertical
                        if (X <= max(line1[x1], line1[x2]) and X >= min(line1[x1],
                                                                        line1[
                                                                            x2])):  # line1 passes through this point
                            if Y <= max(line2[y1], line2[y2]) and Y >= min(line2[y1], line2[
                                y2]):  # line2 also passes through this point
                                points.append((X, Y))
                                intersection_lines.append(line1_copy)
                                intersection_lines.append(line2_copy)


                            elif abs(Y - line2[y1]) < LINE_POINT_Y_DISTANCE or abs(
                                    Y - line2[y2]) < LINE_POINT_Y_DISTANCE:  # Y lies in some distance
                                points.append((X, Y))
                                if line2[y1] > Y:
                                    line2_copy[y1] = Y
                                elif line2[y2] < Y:
                                    line2_copy[y2] = Y
                                intersection_lines.append(line1_copy)
                                intersection_lines.append(line2_copy)

                        elif (Y <= max(line2[y1], line2[y2]) and Y >= min(line2[y1], line2[
                            y2])):  # line2 passes through this point
                            if (X <= max(line1[x1], line1[x2]) and X >= min(line1[x1], line1[
                                x2])):  # line1 also passes through this point
                                points.append((X, Y))
                                intersection_lines.append(line1_copy)
                                intersection_lines.append(line2_copy)

                            elif abs(X - line1[x1]) < LINE_POINT_X_DISTANCE or abs(
                                    X - line1[x2]) < LINE_POINT_X_DISTANCE:  # X lies in some distance
                                points.append((X, Y))
                                if line1[x2] < X:
                                    line1_copy[x2] = X
                                elif line1[x1] > X:
                                    line1_copy[x1] = X
                                intersection_lines.append(line1_copy)
                                intersection_lines.append(line2_copy)

    points = list(set(points))
    unique_data = [list(y) for y in set(tuple(x) for x in intersection_lines)]
    return points, unique_data


def join_vertical_lines(lines):
    import copy
    x1, y1, x2, y2 = 0, 1, 2, 3
    lines = arrange_lines(lines, 0, 1)

    joined_lines = []
    vertical_lines = []
    for line in lines:
        if lineprop.slope(line) > 100:
            vertical_lines.append(line)
        else:
            joined_lines.append(line)

    previous_line = None
    for line in vertical_lines:

        if previous_line is not None and (abs(max(previous_line[y1], previous_line[y2]) - min(line[y1], line[
            y2])) < Y_DEVIATION_VERTCAL_DISTANCE):
            new_line = copy.deepcopy(line)

            new_line[x1] = min(previous_line[x1], previous_line[x2], line[x1], line[x2])
            new_line[y1] = min(previous_line[y1], previous_line[y2], line[y1], line[y2])
            new_line[x2] = max(previous_line[x1], previous_line[x2], line[x1], line[x2])
            new_line[y2] = max(previous_line[y1], previous_line[y2], line[y1], line[y2])

            if lineprop.slope(new_line) > 100:
                line = copy.deepcopy(new_line)
            else:
                joined_lines.append(previous_line)
        elif previous_line is not None:
            joined_lines.append(previous_line)

        previous_line = copy.deepcopy(line)

    joined_lines.append(previous_line)
    return joined_lines



def join_horizontal_lines(lines):
    x1, y1, x2, y2 = 0, 1, 2, 3
    lines = arrange_lines(lines, 1, 0)
    joined_lines = []
    previous_line = ''

    horizontal_lines = []
    for line in lines:
        if lineprop.slope(line) < 1:
            horizontal_lines.append(line)
        else:
            joined_lines.append(line)

    for line in horizontal_lines:
        if type(previous_line) is not str \
                and line[x1] - previous_line[x2] < X_DEVIATION \
                and line[y1] - previous_line[y1] < Y_DEVIATION \
                and (line[y2] - line[y1]) == (previous_line[y2] - previous_line[y1]):
            line[x1] = previous_line[x1]
            line[y1] = previous_line[y1]
        else:
            if type(previous_line) is not str and abs(previous_line[y2] - previous_line[y1]) < Y_DEVIATION:
                joined_lines.append(previous_line)
            elif type(previous_line) is not str and lineprop.slope(previous_line) > 1:
                joined_lines.append(previous_line)
        previous_line = copy.deepcopy(line)

    # final line join
    if abs(previous_line[y2] - previous_line[y1]) < Y_DEVIATION:
        joined_lines.append(previous_line)
    
    elif lineprop.slope(previous_line) > 1:
        joined_lines.append(previous_line)

    joined_lines = arrange_lines(joined_lines, 1, 0)

    return joined_lines


def remove_left_right_borders(lines):
    x1, y1, x2, y2 = 0, 1, 2, 3
    ouput_lines = sorted(lines, key=itemgetter(0, 1))
    max_x = ouput_lines[-1][x2]
    min_x = ouput_lines[0][x1]

    removed_lines = []
    for line in ouput_lines:
        if line[x2] < max_x - BORDER_PIXELS_TO_REMOVE and line[x2] > min_x + BORDER_PIXELS_TO_REMOVE:
            removed_lines.append(line)

    return removed_lines


def join_partial_lines(points, lines):
    x1, y1, x2, y2 = 0, 1, 2, 3

    points = sorted(points, key=itemgetter(1, 0))

    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        if lineprop.slope(line) < 1:
            horizontal_lines.append(line)
        else:
            vertical_lines.append(line)

    new_lines = []
    # join horizontal lines
    previous_point = None
    for point in points:
        if previous_point is not None:
            for line in horizontal_lines:
                if abs(point[y1] - previous_point[y1]) < 10 and line[x1] >= previous_point[x1] and line[x2] <= \
                        point[
                            x1] and (abs(line[y1] - previous_point[y1]) < 10 or abs(line[y2] - point[y1]) < 10):
                    new_lines.append(
                        [int(previous_point[x1]), int(previous_point[y1]), int(point[x1]), int(point[y1])])
                else:
                    pass
        previous_point = point
    lines = lines + new_lines
    return lines





def beam_search(img):
    probable_lines = []
    for y_axis in range(0, len(img)):
        if y_axis < len(img) / 10 or y_axis > len(img) - (len(img) / 10):
            continue
        if sum(img[y_axis]) == len(img[y_axis]) * 255:
            probable_lines.append([0, y_axis, len(img[y_axis] - 1), y_axis])

    if len(probable_lines) > 5:
        probable_lines = []

    return probable_lines

#.........................................OLD MERGE LINES START...................................................

def process_horizontal_lines(horizontal_df, horizontal_df_evidence):
    ################
    # Processing Horizontal lines
    ################
    horizontal_df['ignore'] = 0
    ignore = horizontal_df.columns.get_loc('ignore')
    x2loc = horizontal_df.columns.get_loc('x2')
    horizontal_df = horizontal_df.sort_values('x1')
    horizontal_df.index = range(len(horizontal_df))
    # horizontal_df.head()
    for index, row in horizontal_df.iterrows():
        if row.ignore == 1:
            continue
        temp = horizontal_df[
            (horizontal_df.y1 > row.y1 - 15) & (horizontal_df.y1 < row.y1 + 15) & (horizontal_df.ignore == 0)]
        temp = temp.sort_values(['x1'])
        temp['index'] = temp.index.tolist()
        temp.index = range(len(temp))
        dontextend = False
        for _, row1 in temp.iterrows():

            # check if this is part of some line in temp
            if (index == row1['index']): continue
            if ((row1.x1 <= row.x1 + 5) & (row1.x2 >= row.x2 - 5)):
                dontextend = True
                horizontal_df.iloc[index, ignore] = 1
                break
        if dontextend: continue
        # extend 60, 30, 15 pixels and exit by seeing evidence from run3
        previous = row.x2
        rowx2 = row.x2
        while (True):
            rowx2 += 5
            # check for the evidence in horizontal_df_evidence
            if len(horizontal_df_evidence[
                       (horizontal_df_evidence.y1 > row.y1 - 10) & (horizontal_df_evidence.y1 < row.y1 + 10) & (
                               horizontal_df_evidence.x1 <= previous) & (horizontal_df_evidence.x2 >= rowx2)]) > 0:
                previous = rowx2
                continue
            else:
                rowx2 -= 5
                break
        horizontal_df.iloc[index, x2loc] = rowx2
    return horizontal_df


def process_vertical_lines(vertical_df, vertical_df_evidence):
    ################
    # Processing Horizontal lines
    ################
    vertical_df['ignore'] = 0
    ignore = vertical_df.columns.get_loc('ignore')
    y2loc = vertical_df.columns.get_loc('y2')
    vertical_df = vertical_df.sort_values('y1')
    vertical_df.index = range(len(vertical_df))

    for index, row in vertical_df.iterrows():
        if row.ignore == 1:
            continue
        temp = vertical_df[(vertical_df.x1 > row.x1 - 15) & (vertical_df.x1 < row.x1 + 15) & (
                vertical_df.ignore == 0)]  # & (vertical_df.y2 < row.y2 + 15)
        temp = temp.sort_values(['y1'])
        temp['index'] = temp.index.tolist()
        temp.index = range(len(temp))
        dontextend = False
        for _, row1 in temp.iterrows():

            # check if this is part of some line in temp
            if (index == row1['index']): continue
            if ((row1.y1 <= row.y1) & (row1.y2 >= row.y2)):
                dontextend = True
                vertical_df.iloc[index, ignore] = 1
                break
        if dontextend: continue

        previous = row.y2
        rowy2 = row.y2

        # extend 60, 30, 15 pixels and exit by seeing evidence from run3
        while (True):
            rowy2 += 10
            # check for the evidence in vertical_df_evidence
            if len(vertical_df_evidence[
                       (vertical_df_evidence.x1 > row.x1 - 5) & (vertical_df_evidence.x1 < row.x1 + 5) & (
                               vertical_df_evidence.y1 <= previous + 3) & (
                               vertical_df_evidence.y2 >= rowy2 - 3)]) > 0:
                previous = rowy2
                continue
            else:
                rowy2 -= 10
                break

        vertical_df.iloc[index, y2loc] = rowy2

    return vertical_df


def preprocess_df(data):
    data['xdiff'] = abs(data['x1'] - data['x2'])
    data['ydiff'] = abs(data['y1'] - data['y2'])
    # find those lines which are neither parallel to x-axis nor y-axis and fix them
    tempdata = data[(data['x1'] != data['x2']) & (data['y1'] != data['y2'])]
    data = data.sort_values(['x1'])

    for index, row in tempdata.iterrows():
        if (row['xdiff'] < row['ydiff']):
            data.iloc[index, data.columns.get_loc('x2')] = row['x1']
        else:
            data.iloc[index, data.columns.get_loc('y2')] = row['y1']

    # split the data into vertical_lines and horizontal_lines lines
    horizontal_lines = data[data['y1'] == data['y2']]
    horizontal_lines = horizontal_lines.drop(['ydiff'], axis=1)
    horizontal_lines = horizontal_lines.sort_values('xdiff')
    horizontal_lines.index = range(len(horizontal_lines))
    vertical_lines = data[data['x1'] == data['x2']]
    vertical_lines = vertical_lines.drop(['xdiff'], axis=1)
    vertical_lines = vertical_lines.sort_values('ydiff')
    vertical_lines.index = range(len(vertical_lines))

    # Make x1 < x2 for vertical_lines lines
    temp = horizontal_lines[horizontal_lines.x2 < horizontal_lines.x1]
    if len(temp) > 0:
        x1coord = horizontal_lines.columns.get_loc('x1')
        x2coord = horizontal_lines.columns.get_loc('x2')
        for index, row in temp.iterrows():
            horizontal_lines.iloc[index, x1coord] = row['x2']
            horizontal_lines.iloc[index, x2coord] = row['x1']

    # Make y1 < y2 for vertical_lines lines
    temp = vertical_lines[vertical_lines.y2 < vertical_lines.y1]
    if len(temp) > 0:
        y1coord = vertical_lines.columns.get_loc('y1')
        y2coord = vertical_lines.columns.get_loc('y2')
        for index, row in temp.iterrows():
            vertical_lines.iloc[index, y1coord] = row['y2']
            vertical_lines.iloc[index, y2coord] = row['y1']

    return horizontal_lines, vertical_lines


def merge_lines(line1, line3):
    # Create DataFrames
    data1 = pd.DataFrame(line1, columns=['x1', 'y1', 'x2', 'y2'])
    data3 = pd.DataFrame(line3, columns=['x1', 'y1', 'x2', 'y2'])

    # preprcess dataframes to arrange coordinates and make perfect vertical / horizontal
    horizontal_lines_support, vertical_lines_support = preprocess_df(data1)
    horizontal_lines_final, vertical_lines_final = preprocess_df(data3)
    horizontal = process_horizontal_lines(horizontal_lines_support, horizontal_lines_final)
    vertical = process_vertical_lines(vertical_lines_support, vertical_lines_final)
    # update temp as well as horizontal1
    #################
    ##Desired output#
    #################
    horizontal = horizontal[horizontal.ignore == 0]
    horizontal = horizontal[['x1', 'y1', 'x2', 'y2']]
    vertical = vertical[vertical.ignore == 0]
    vertical = vertical[['x1', 'y1', 'x2', 'y2']]
    lines = pd.concat([vertical, horizontal], ignore_index=True)
    lines = lines.values.tolist()

    return lines

def remove_overlaplines(lines):
   lines_overlap = []
   indexes_overlap = []
   for index_x, line_x in enumerate(lines):
       for index_y, line_y in enumerate(lines):
           if index_y != index_x:
               if lineprop.intersection_financial_lines(line_x, line_y):

                   if (index_y not in indexes_overlap) and (index_x not in indexes_overlap):
                       lines_overlap.append(
                           [min(line_x[0], line_y[0]), int((line_x[1] + line_y[1]) / 2), max(line_x[2], line_y[2]),
                            int((line_x[3] + line_y[3]) / 2)])
                       indexes_overlap.append(int(index_y))
                       indexes_overlap.append(int(index_x))
   indexes_overlap = list(set(indexes_overlap))
   indexes_overlap = sorted(indexes_overlap)
   lines_remove = []
   for index, point in enumerate(lines):
       if index not in indexes_overlap:
           lines_remove.append(lines[index])
   lines_new = lines_remove + lines_overlap
   return lines_new