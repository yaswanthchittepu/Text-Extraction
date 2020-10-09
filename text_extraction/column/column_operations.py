import pandas as pd

def get_headers(colboxes):
    top_left_x = []
    top_left_y = []
    bottom_right_x = []
    bottom_right_y = []
    box_mid_x = []
    for row_no in range(0, len(colboxes)):
        top_left_x.append(colboxes[row_no][0])
        top_left_y.append(colboxes[row_no][1])
        bottom_right_x.append(colboxes[row_no][2])
        bottom_right_y.append(colboxes[row_no][3])
        box_mid_x.append(int((colboxes[row_no][1] + colboxes[row_no][3]) / 2))
    boxes_points = pd.DataFrame()
    boxes_points['x1'] = top_left_x
    boxes_points['y1'] = top_left_y
    boxes_points['x2'] = bottom_right_x
    boxes_points['y2'] = bottom_right_y
    boxes_points['mid'] = box_mid_x
    boxes = []

    start_index = 0
    next_index = 1

    indices = []
    while (next_index != len(colboxes)):
        if abs(boxes_points.iloc[start_index]['mid'] - boxes_points.iloc[next_index]['mid']) <= 70:
            next_index = next_index + 1
        else:
            indices.append(start_index)
            start_index = next_index
            next_index = next_index + 1
    indices.append(start_index)

    for index in range(0, len(indices)):
        if index != len(indices) - 1:
            header_box_filter = boxes_points[indices[index]:indices[index + 1]]

            header_box_filter = header_box_filter.sort_values(['x1'])
            boxes.append([80, int(header_box_filter.iloc[0]['y1']), int(header_box_filter.iloc[0]['x1'] - 30), int(header_box_filter.iloc[0]['y2'])])
        else:
            header_box_filter = boxes_points[indices[index]:]
            header_box_filter = header_box_filter.sort_values(['x1'])

            boxes.append([80, int(header_box_filter.iloc[0]['y1']), int(header_box_filter.iloc[0]['x1'] - 30), int(header_box_filter.iloc[0]['y2'])])

    return boxes

def get_column_boxes(uniq_groups, column_lines, img_master):

    color_groups = [[0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [100, 255, 100]]

    bounding_boxes = []

    for index, group in enumerate(uniq_groups):
        color = color_groups[0]
        column_frame = column_lines[column_lines['cluster_group'] == group]
        if (len(column_frame) > 2) and (column_frame['linelength'].max() < 250):
            column_frame = column_frame.sort_values(['y1'])
            n = len(column_frame)
            if (column_frame.iloc[0]['linelength'] > 30) and (column_frame.iloc[n - 1]['linelength'] > 30) and (
                    int(column_frame.iloc[n - 1]['x2']) - int(column_frame.iloc[0]['x1']) > 30):

                if n > 4:
                    bounding_boxes.append(
                        [int(column_frame.iloc[0]['x1']), int(column_frame.iloc[0]['y1'] - 60), int(column_frame.iloc[n - 1]['x2']),
                         int(column_frame.iloc[n - 1]['y2'])])
                if n < 4:
                    bounding_boxes.append(
                        [int(column_frame.iloc[0]['x1']), int(column_frame.iloc[0]['y1'] - 100), int(column_frame.iloc[n - 1]['x2']),
                         int(column_frame.iloc[n - 1]['y2'])])
    header_box = []

    return bounding_boxes, header_box

def find_mini_box(column_lines):
    uniq_groups = column_lines['cluster_group'].unique()
    minibox = pd.DataFrame()
    big_boxes = []
    boxes = []
    try:
        for group in uniq_groups:
            column_data = column_lines[column_lines['cluster_group'] == group]
            if len(column_data) == 2:
                for search_group in uniq_groups:
                    group_data = column_lines[column_lines['cluster_group'] == search_group]
                    if len(group_data) < 10 and (group != search_group) and len(group_data) > 1:
                        if group_data.iloc[0]['y1'] > column_data.iloc[0]['y1'] and group_data.iloc[len(group_data) - 1][
                            'y1'] < column_data.iloc[1]['y1']:
                            minibox = pd.concat([minibox, group_data])
                big_boxes.append(
                    [int(80), int(column_data.iloc[0]['y1'] - 70), int(column_data.iloc[1]['x1'] - 20), int(column_data.iloc[1]['y1'])])

        minigroups = minibox['cluster_group'].unique()

        for mini in minigroups:
            small_boxes = minibox[minibox['cluster_group'] == mini]
            boxes.append([int(small_boxes.iloc[0]['x1']), int(small_boxes.iloc[0]['y1'] - 120),
                          int(small_boxes.iloc[len(small_boxes) - 1]['x2']),
                          int(small_boxes.iloc[len(small_boxes) - 1]['y2'] + 30)])
    except:
        return boxes
    return boxes + big_boxes
