import numpy as np
from operator import itemgetter
from apporchid.common.logger import logger as logger
import apporchid.agora.cv.line.line_detection as line_detection
import apporchid.agora.cv.line.line_operations as lineop


# def detect_points(lines):
#     points_lst, new_lines_lst = [], []
#     for lines in lines:
#         points, new_lines = [], []
#         try:
#             points, new_lines = line_detection.find_points_of_intersection_and_extend_lines(lines)
#             points = remove_adjacent_points(points)
#         except Exception as e:
#             logger.exception(e)
#         if (len(points) != 0):
#             points_lst.append(points)
#         if (len(new_lines) != 0):
#             new_lines_lst.append(new_lines)
#     points = points_lst
#     lines = new_lines_lst
#     return points, lines

def detect_points(lines):
    logger.debug('-------------------------------------------------')
    logger.debug('Starting Point detection for : {0}'.format(lines))
    points = []
    new_lines = []
    try:
        points, new_lines = lineop.find_points_of_intersection_and_extend_lines(lines)
        points = remove_adjacent_points(points)
    except Exception as e:
        logger.exception(e)
    return points, new_lines


def detect_points_in_image_lines(lines):
    points_lst, new_lines_lst = [], []
    for lines in lines:
        points, new_lines = detect_points(lines)
        points_lst.append(points)
        new_lines_lst.append(new_lines)
        logger.debug('-------------------------------------------------')
        logger.debug('Starting Point detection for : {0}'.format(lines))
    points = points_lst
    lines = new_lines_lst
    return points, lines


def remove_adjacent_points(points):
    x, y = 0, 1

    # sort an remove points vertically
    points = sorted(points, key=itemgetter(x, y))
    previous_point = ''

    final_points = []
    for point in points:
        if type(previous_point) is not str and abs(previous_point[x] - point[x]) < 5 and abs(
                previous_point[y] - point[y]) < 10:
            pass

        elif type(previous_point) is not str:
            final_points.append(previous_point)
            previous_point = point

        if type(previous_point) is str:
            previous_point = point

    if type(previous_point) is not str:
        final_points.append(previous_point)

    # sort and remove point horizontally
    points = sorted(final_points, key=itemgetter(y, x))

    previous_point = ''

    final_points = []
    for point in points:
        if type(previous_point) is not str and abs(previous_point[x] - point[x]) < 20 and abs(
                previous_point[y] - point[y]) < 5:
            pass
        elif type(previous_point) is not str:
            final_points.append(previous_point)
            previous_point = point
        if type(previous_point) is str:
            previous_point = point

    if type(previous_point) is not str:
        final_points.append(previous_point)

    # commenting out below section for diagnoal sort for time being
    # add cartesean distance for sorting

    distance_appended_points = []
    for point in points:
        point = list(point)
        point.append(np.sqrt(np.power(point[x], 2) + np.power(point[y], 2)))
        distance_appended_points.append(point)

    #sort and remove point horizontally by cartesean distance
    points = sorted(distance_appended_points, key=itemgetter(2))

    previous_point = ''

    final_points = []
    for point in points:
        point = tuple([point[x], point[y]])
        if type(previous_point) is not str and abs(previous_point[x] - point[x]) < 10 and abs(
                previous_point[y] - point[y]) < 10:
            pass
        elif type(previous_point) is not str:
            final_points.append(previous_point)
            previous_point = point
        if type(previous_point) is str:
            previous_point = point

    if type(previous_point) is not str:
        final_points.append(previous_point)

    return final_points
