import apporchid.agora.cv.image.draw_on_image as draw
import apporchid.agora.cv.image.image_loader as image_loader
import apporchid.agora.cv.image.image_cleaner as image_cleaner
import apporchid.agora.cv.image.image_operations as image_operations
import apporchid.agora.cv.line.line_detection as line_detection
import apporchid.agora.cv.line.line_operations as lineop
import apporchid.agora.cv.column.column_operations as colops
import apporchid.agora.cv.column.column_properties as colprop
import apporchid.common.utils as storage
import cv2
from collections import Counter

import numpy as np

import apporchid.agora.cv.word.word_bounding_box as word_contour_detection

import apporchid.agora.cv.column.bounding_box_operations as bbops
from apporchid.agora.cv.image.draw_on_image import draw_lines

def get_boxes_for_line_format(file, input_dir, output_dir):
    file_name = '.'.join(file.split('.')[:-1])
    # Get the image
    img = image_loader.get_image(input_dir, file)

    # Clean the image by removing shades
    #clean_image = image_cleaner.remove_shades(img, True, file_name)

    clean_image = image_operations.convert_to_gray(img)

    #gray_image = image_operations.convert_to_gray(img)
    #cv2.imwrite('C:/Workspace/PyWorkspace/FinancialDocs/apporchid/financialdocs/debug/' + 'gray_image.png', gray_image)
    #bw = cv2.adaptiveThreshold(~gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    _,bw = cv2.threshold(~clean_image,100,255,cv2.THRESH_BINARY)

    cv2.imwrite('C:/Workspace/PyWorkspace/FinancialDocs/apporchid/financialdocs/debug/' + 'test_bw.png', bw)
    #draw.draw_lines(img, lines_new, output_dir, file)
    
    # Split the image into seprate table images

    roi_lines = line_detection.detect_lines(bw, [], img)

    draw.draw_lines(img, roi_lines, 'C:/Workspace/PyWorkspace/FinancialDocs/apporchid/financialdocs/debug/', file+'roi_lines.png')
   
    # Removing overlapping lines
    lines_new = lineop.remove_overlaplines(roi_lines)

    # # Removing vertical lines
    #lines_new = lineop.remove_short_vertical_lines(lines_new, 1000)

    draw.draw_lines(img, lines_new, 'C:/Workspace/PyWorkspace/FinancialDocs/apporchid/financialdocs/debug/', file+'remove_overlaplines.png')

    # Storing line coordinates to dataframe
    column_line_boxes = storage.store_lines_to_dataframe(lines_new)

    # Arranging lines based on y-coordinate
    column_line_boxes = column_line_boxes.sort_values(['y1'])

    # generating midpoint of lines
    column_line_boxes = colprop.store_midpoints_for_line(column_line_boxes)

    # grouping the line to clusters
    column_line_boxes = colprop.cluster_to_groups(column_line_boxes)

    uniq_groups = column_line_boxes['cluster_group'].unique()

    # Getting the columns based on line clusters
    bounding_boxes, header_box = colops.get_column_boxes(uniq_groups, column_line_boxes, clean_image)

    # header_box = colops.get_headers(bounding_boxes)
    mini_boxes = []
    header_box = []
    new_boxes = bounding_boxes + header_box
    new_boxes = mini_boxes + new_boxes
    return new_boxes

def bounding_boxes_withoutlines(file, input_dir, output_dir):

    image = cv2.imread(input_dir+file)

    blank_image = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # dilation
    kernel = np.ones((5, 15), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    # find contours
    rectangle_coordinates = word_contour_detection.word_box_contours(img_dilation, image)

    # storing the word level bounding boxes to dataframe
    column_lines_frame = storage.store_rectangle_points_to_dataframe(rectangle_coordinates)

    # Clustering the columns and assigning it to a group
    column_lines_frame = bbops.cluster_boundingboxes_to_groups(column_lines_frame)

    # getting the bounding boxes of each group that is clustered
    boxes = bbops.get_bounding_boxes_for_columns_from_word_level(column_lines_frame)

    # making uneven boxes to even boxes
    boxes = bbops.uneven_columns_to_even_columns(boxes)

    groups = bbops.singular_clusters(Counter(column_lines_frame['group'].tolist()))

    # remaining groups that are left out of clusters
    word_boxes_left = column_lines_frame[column_lines_frame['group'].isin(groups)]

    # calculating the midpoints fo rectangular boxes that are remaining
    word_boxes_left = bbops.midpoint_of_bounding_boxes(word_boxes_left)

    bounding_boxes_data = storage.store_rectangle_points_to_dataframe(boxes)
    bounding_boxes_data = bounding_boxes_data.sort_values(['x1']).reset_index()

    # midpoint of dataframe
    bounding_boxes_data = bbops.midpoint_of_bounding_boxes(bounding_boxes_data)

    bounding_boxes, word_boxes = bbops.merge_bounding_boxes(bounding_boxes_data, word_boxes_left)

    # get bounding column boxes
    boxes = []
    for box_index in range(0, len(bounding_boxes)):
        word_box_index = bounding_boxes.iloc[box_index]['index']
        if word_box_index != 0:
            if word_boxes.loc[word_box_index]['index'] == 1:
                if word_boxes.loc[word_box_index]['x1'] > bounding_boxes.iloc[box_index]['x2']:
                    topleft_x, topleft_y = bounding_boxes.iloc[box_index]['x1'], word_boxes.loc[word_box_index]['y1']
                    bottomright_x, bottomright_y = word_boxes.loc[word_box_index]['x2'], bounding_boxes.iloc[box_index]['y2']
                    boxes.append([topleft_x, topleft_y, bottomright_x, bottomright_y])
                else:
                    topleft_x, topleft_y = word_boxes.loc[word_box_index]['x1'], word_boxes.loc[word_box_index]['y1']
                    bottomright_x, bottomright_y = bounding_boxes.iloc[box_index]['x2'], bounding_boxes.iloc[box_index]['y2']
                    boxes.append([topleft_x, topleft_y, bottomright_x, bottomright_y])

            else:
                topleft_x, topleft_y = bounding_boxes.iloc[box_index]['x1'], word_boxes.loc[word_box_index]['y1']
                bottomright_x, bottomright_y = word_boxes.loc[word_box_index]['x2'], bounding_boxes.iloc[box_index]['y2']
                boxes.append([topleft_x, topleft_y, bottomright_x, bottomright_y])

        else:
            bottomright_x, bottomright_y = bounding_boxes.iloc[box_index]['x2'], bounding_boxes.iloc[box_index]['y2']
            topleft_x, topleft_y = bounding_boxes.iloc[box_index]['x1'], bounding_boxes.iloc[box_index]['y1']
            boxes.append([topleft_x, topleft_y, bottomright_x, bottomright_y])


    boxes = bbops.uneven_columns_to_even_columns(boxes)


    return boxes

def get_lines_from_gray_image(file, input_dir, output_dir):
    import cv2
    image = cv2.imread(input_dir+file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[gray < 220] = 0
    gray_sum = gray.sum(axis=1)
    lines = []
    th = len(gray[0]) * 255 * 0.2
    white = True
    for i in range(len(gray_sum)):
        if white == True and gray_sum[i] >= th:
            continue
        elif white == True and gray_sum[i] <= th:
            white = False
            lines.append(i)
        elif white == False and gray_sum[i] <= th:
            continue
        else:
            white = True
            lines.append(i)

    return lines