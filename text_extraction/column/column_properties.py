from sympy.geometry import Point

def cluster_to_groups(column_lines):
    group = 1
    groupupdated = []
    large_line_filter = 0
    for index_first in range(len(column_lines)):
        groupupdated = list(column_lines['cluster_group'].unique())
        if len(groupupdated) > 1:
            try:
                groupupdated.remove(0)
            except:
                pass
        else:
            groupupdated = []

        groupupdated = [grouped for grouped in groupupdated if grouped is not None]
        if column_lines.iloc[index_first]['cluster_group'] not in groupupdated:
            for index_second in range(len(column_lines)):
                if (abs(column_lines.iloc[index_second]['x_midpoint'] - column_lines.iloc[index_first]['x_midpoint']) <= 10) and index_first != index_second and index_first < index_second:
                    if (abs(column_lines.iloc[index_first]['x1'] - column_lines.iloc[index_second]['x1']) <= 40) & (
                            abs(column_lines.iloc[index_first]['x2'] - column_lines.iloc[index_second]['x2']) <= 40):
                        if large_line_check(column_lines, index_first, index_second):
                            large_line_filter = 1
                            break
                        else:
                            column_lines.iloc[index_second, column_lines.columns.get_loc('cluster_group')] = group

                            column_lines.iloc[index_first, column_lines.columns.get_loc('cluster_group')] = group
                if large_line_filter == 1:
                    large_line_filter = 0
        group = group + 1
    return column_lines

def large_line_check(column_lines, a_i, a_j):

    for a_k in range(a_i, a_j):
        if (abs(column_lines.iloc[a_i]['linelength'] - column_lines.iloc[a_k]['linelength']) > 100) and (
                column_lines.iloc[a_i]['x1'] > column_lines.iloc[a_k]['x1']) and (column_lines.iloc[a_i]['x1'] < column_lines.iloc[a_k]['x2']) and (
        abs(column_lines.iloc[a_i]['x2'] - column_lines.iloc[a_k]['x2'] < 10)):
            return 1
    return 0


def store_midpoints_for_line(column_lines):
    column_lines['cluster_group'] = 0
    column_lines['linelength'] = 0
    for index in range(len(column_lines)):
        p1, p2 = Point(column_lines.iloc[index]['x1'], column_lines.iloc[index]['y1']), Point(column_lines.iloc[index]['x2'], column_lines.iloc[index]['y2'])
        column_lines.iloc[index, column_lines.columns.get_loc('linelength')] = p1.distance(p2)
        column_lines['x_midpoint'] = (column_lines['x1'] + column_lines['x2']) / 2.0
        column_lines['y_midpoint'] = (column_lines['y1'] + column_lines['y2']) / 2.0
    return column_lines

def eliminating_uneven_boxes(bounding_boxes):
    index = 0
    outlier = 0
    counts = 0
    new_val = 0
    for top_left_x, top_left_y, right_bottom_x, right_bottom_y in bounding_boxes:
        if index == 0:
            y_min = top_left_y
        if not (abs(y_min - top_left_y) < 7):
            outlier = top_left_y
            new_val = y_min
            counts = counts + 1
        index = index + 1

    new_boxes = []

    if counts > 1:
        outlier = y_min
        new_val = top_left_y

    for top_left_x, top_left_y, right_bottom_x, right_bottom_y in bounding_boxes:
        if top_left_y == outlier:
            new_boxes.append([top_left_x, new_val, right_bottom_x, right_bottom_y])
        else:
            new_boxes.append([top_left_x, top_left_y, right_bottom_x, right_bottom_y])
    return new_boxes