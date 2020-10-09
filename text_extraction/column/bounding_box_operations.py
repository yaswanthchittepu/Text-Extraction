import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance


def merge_bounding_boxes(boxes, word_boxes):

    boxes_combined = []
    boxes_indices = boxes.index.values
    boxes['index'] = 0
    word_boxes_indices = word_boxes.index.values
    prev_p = 0
    prev_q = 0
    word_boxes['index'] = 0
    last_y = 0

    for box_index in boxes_indices[1:]:
        min_value = 9999999
        x_min_value = 99999999
        word_index_select = -1
        index_selected = -1

        for word_index in word_boxes_indices:
            if not boxes.loc[box_index]['index'] and not word_boxes.loc[word_index]['index']:
                x_value = boxes.loc[box_index]['xmid']
                y_value = boxes.loc[box_index]['y1']
                x1_value = word_boxes.loc[word_index]['xmid']
                y1_value = word_boxes.loc[word_index]['ymid']
                column_box_tuple = (x_value, y_value)
                word_box_tuple = (x1_value, y1_value)
                dst = abs(y_value - y1_value)
                x_dst = abs(x_value - x1_value)
                if y_value > y1_value and abs(y_value - y1_value) < 12 and x_dst < x_min_value:
                    min_value = dst
                    x_min_value = x_dst
                    word_index_select = box_index
                    index_selected = word_index
                    last_y = y_value

        if word_index_select != -1 and index_selected != -1:
            boxes.set_value(word_index_select, 'index', index_selected)
            word_boxes.set_value(index_selected, 'index', 1)

    min_value = 9999999

    for header_box in [boxes_indices[0]]:
        for word_box_index in word_boxes_indices:
            if not boxes.loc[header_box]['index'] and not word_boxes.loc[word_box_index]['index']:
                x_value = boxes.loc[header_box]['xmid']
                y_value = boxes.loc[header_box]['y1']
                x1_value = word_boxes.loc[word_box_index]['xmid']
                y1_value = word_boxes.loc[word_box_index]['ymid']
                word_box_tuple = (x_value, y_value)
                column_box_tuple = (x1_value, y1_value)
                dst = distance.euclidean(word_box_tuple, column_box_tuple)
                if dst < min_value and y_value > y1_value and abs(y_value - y1_value) < 12:
                    min_value = dst
                    word_index_select = header_box
                    index_selected = word_box_index
    if word_index_select != -1 and index_selected != -1 and not word_boxes.loc[index_selected]['index']:
        boxes.set_value(word_index_select, 'index', index_selected)
        if abs(last_y - y_value) > 20:
            word_boxes.set_value(index_selected, 'index', -1)
        else:
            word_boxes.set_value(index_selected, 'index', 1)

    return boxes, word_boxes


def midpoint_of_bounding_boxes(rectangular_points):
    rectangular_points['xmid'] = (rectangular_points['x1'] + rectangular_points['x2']) / 2

    rectangular_points['ymid'] = (rectangular_points['y1'] + rectangular_points['y2']) / 2

    rectangular_points['xmid'] = rectangular_points['xmid'].astype(int)

    rectangular_points['ymid'] = rectangular_points['ymid'].astype(int)

    return rectangular_points


def singular_clusters(clusters):
    groupids = []

    for key, value in clusters.items():
        if value < 3:
            groupids.append(key)

    return groupids


def uneven_columns_to_even_columns(boxes):
    y1_min = 10000

    y2_max = 0

    new_boxes = []

    for x1, y1, x2, y2 in boxes:
        if y1 < y1_min:
            y1_min = y1
        if y2 > y2_max:
            y2_max = y2

    for box in boxes:
        new_boxes.append((box[0], y1_min, box[2], y2_max))

    return new_boxes


def get_bounding_boxes_for_columns_from_word_level(wordlevel_boxes_with_group):
    clusters = wordlevel_boxes_with_group['group'].unique()

    boxes = []

    for group in clusters:
        column_cluster = wordlevel_boxes_with_group[wordlevel_boxes_with_group['group'] == group]
        if len(column_cluster) >= 3:
            column_cluster = column_cluster.sort_values(['y1'])
            if len(column_cluster[abs(column_cluster['x1'] - column_cluster['x2']) > 500]):
                second_box = column_cluster['x2'].tolist()
                x2_max = np.partition(second_box, -2)[-2]
            else:
                x2_max = column_cluster['x2'].max()
            y2_max = column_cluster['y2'].max()
            x1_min = column_cluster['x1'].min()
            y1_min = column_cluster['y1'].min()
            boxes.append((x1_min, y1_min, x2_max, y2_max))

    return list(set(boxes))


def cluster_boundingboxes_to_groups(frame):

    frame_copy = frame

    frame = frame_copy[(frame_copy['y2'] < 590) | (frame_copy['x2'] < 760)]

    group_length = len(frame)

    row = 0
    boxes_per_cluster = pd.DataFrame()
    cluster_group = 0
    boolean_check = 1

    while group_length:
        if boolean_check:
            intermediate = frame[abs(frame_copy.iloc[row]['x1'] - frame['x1']) < 3]
            if len(intermediate) > 7:
                boolean_check = 0
        else:
            intermediate = frame[abs(frame_copy.iloc[row]['x2'] - frame['x2']) < 3]

        s1 = pd.merge(frame, intermediate, how='inner', on=['x1', 'y1', 'x2', 'y2'])

        frame = pd.concat([frame, s1, s1]).drop_duplicates(keep=False)
        s1['group'] = cluster_group
        cluster_group = cluster_group + 1
        group_length = len(frame)
        frame_copy = frame
        boxes_per_cluster = pd.concat([boxes_per_cluster, s1])

    boxes_per_cluster = boxes_per_cluster.reset_index()

    return boxes_per_cluster


def remove_shades(img, normalize=True, fname_img='file1'):
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []

    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((3, 3), np.uint8))
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
    cv2.imwrite(fname_img + '_ShadeRemoval.png', result)

    return result
