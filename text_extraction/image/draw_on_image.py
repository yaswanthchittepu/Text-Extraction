import cv2
import os
import random
from apporchid.common.logger import logger

def draw_boxes(img_master, new_boxes, image_output_dir, file):

    for top_left_x, top_left_y, bottom_right_x, bottom_right_y in new_boxes:
        cv2.rectangle(img_master, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), (0, 255, 0), 3)
    cv2.imwrite(os.path.join(image_output_dir, file), img_master)


def draw_lines(img_master, new_boxes, image_output_dir, file):

    img_master1 = img_master.copy()
    for box  in new_boxes:
        left_extreme_x, left_extreme_y, right_extreme_x, right_extreme_y = box
        cv2.line(img_master1, (int(left_extreme_x), int(left_extreme_y)), (int(right_extreme_x), int(right_extreme_y)), (0, 255, 0), 3)
    cv2.imwrite(os.path.join(image_output_dir, 'line_write__' + file), img_master1)


def draw_column_lines(uniq_groups, boxes, img_master):

    color_groups = [[0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [100, 255, 100]]
    bounding_boxes = []

    for indexes, group in enumerate(uniq_groups):
        color1 = random.randint(0, 255)
        color2 = random.randint(0, 255)
        color3 = random.randint(0, 255)
        cluster_group = boxes[boxes['cluster_group'] == group]

        if (len(cluster_group) > 2) and (cluster_group['linelength'].max() < 200):
            cluster_group = cluster_group.sort_values(['y1'])
            cluster_length = len(cluster_group)
            for index in range(0, cluster_length):
                cv2.line(img_master, (int(cluster_group.iloc[index]['x1']), int(cluster_group.iloc[index]['y1'])),
                         (int(cluster_group.iloc[index]['x2']), int(cluster_group.iloc[index]['y2'])), (color1, color2, color3), 3)


    return 1


def draw_table_to_image(img, lines_snip, points_snip):
    orig_img_copy = img.copy()
    X, Y = [], []
    min_coords = None
    try:
        for l in lines_snip:
            continue
            x1, y1, x2, y2 = l
            cv2.line(orig_img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Write points to output Image
        for i in points_snip:
            cx, cy = i
            X.append(cx)
            Y.append(cy)
            continue
            cv2.circle(orig_img_copy, (int(cx), int(cy)), 10, (255, 255, 255), -11)
            cv2.circle(orig_img_copy, (int(cx), int(cy)), 11, (0, 0, 255), 1)  # draw circle
            cv2.ellipse(orig_img_copy, (int(cx), int(cy)), (10, 10), 0, 0, 90, (0, 0, 255), -1)
            cv2.ellipse(orig_img_copy, (int(cx), int(cy)), (10, 10), 0, 180, 270, (0, 0, 255), -1)
            cv2.circle(orig_img_copy, (int(cx), int(cy)), 1, (0, 255, 0), 1)  # draw center
        x_min, y_min = min(X), min(Y)
        min_coords = (x_min, y_min)
    except Exception as e:
        logger.debug(e)
    return orig_img_copy, min_coords