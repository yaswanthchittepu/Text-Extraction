import cv2
import numpy as np
import apporchid.agora.cv.line.line_detection as line_detection
import apporchid.agora.cv.point.point_detection as point_detection
from apporchid.agora.cv.table.table_json_generator import as_json
import pandas as pd
from apporchid.common.logger import logger as logger
from apporchid.common.config import cfg
import apporchid.agora.cv.image.image_loader as image_loader
import apporchid.agora.cv.image.image_cleaner as image_cleaner
import apporchid.agora.cv.table.table_image_splitter as table_image_splitter
import apporchid.agora.cv.image.draw_on_image as draw
import apporchid.agora.cv.line.line_properties as lineprop

PROJECT = cfg['project']
SUBPROJECT = cfg['subproject']
output_dir = cfg[PROJECT][SUBPROJECT]['output-dir']

def detect_table(snip, bbox_lines, img_to_write, orig_img, leptonica_boxes, output_image_name):

    img_to_write_cpy = img_to_write.copy()

    roi_lines = line_detection.detect_lines(snip, bbox_lines, orig_img)
    roi_points, roi_lines = point_detection.detect_points(roi_lines)


    tableslist, boxlist, img_to_write = _create_table_from_lines(roi_lines, leptonica_boxes, img_to_write_cpy,
                                                                 output_image_name)  # chk
    single_table_json = as_json(tableslist, boxlist)
    # single_table_json = _generate_json(roi_lines)
    img_to_write, min_coords = draw.draw_table_to_image(img_to_write, roi_lines, roi_points)

    return single_table_json, img_to_write, min_coords, tableslist


def _create_table_from_lines(lines, leptonica_boxes, img_to_write_cpy, output_image_name):  # chk

    lines_df = pd.DataFrame(lines, columns=['x1', 'y1', 'x2', 'y2'])

    lines_df['xdiff'] = abs(lines_df['x1'] - lines_df['x2'])
    lines_df['ydiff'] = abs(lines_df['y1'] - lines_df['y2'])

    ###find those lines which are neither parallel to x-axis nor y-axis and fix them
    tempdata = lines_df[(lines_df['x1'] != lines_df['x2']) & (lines_df['y1'] != lines_df['y2'])]
    for index, row in tempdata.iterrows():
        if (row['xdiff'] > 2 and row['ydiff'] > 2):
            continue
        if (row['xdiff'] < row['ydiff']):
            lines_df.iloc[index, lines_df.columns.get_loc('x2')] = row['x1']
        else:
            lines_df.iloc[index, lines_df.columns.get_loc('y2')] = row['y1']

    # split the data into vertical and horizontal lines
    horizontal = lines_df[lines_df['y1'] == lines_df['y2']]
    horizontal = horizontal.drop(['ydiff'], axis=1)
    horizontal = horizontal.sort_values('xdiff')
    horizontal.index = range(len(horizontal))
    vertical = lines_df[lines_df['x1'] == lines_df['x2']]
    vertical = vertical.drop(['xdiff'], axis=1)
    vertical = vertical.sort_values('ydiff')
    vertical.index = range(len(vertical))

    # Make x1 < x2 for vertical lines
    temp = horizontal[horizontal.x2 < horizontal.x1]
    if len(temp) > 0:
        x1coord = horizontal.columns.get_loc('x1')
        x2coord = horizontal.columns.get_loc('x2')
        for index, row in temp.iterrows():
            horizontal.iloc[index, x1coord] = row['x2']
            horizontal.iloc[index, x2coord] = row['x1']
    # Make y1 < y2 for vertical lines
    temp = vertical[vertical.y2 < vertical.y1]
    if len(temp) > 0:
        y1coord = vertical.columns.get_loc('y1')
        y2coord = vertical.columns.get_loc('y2')
        for index, row in temp.iterrows():
            vertical.iloc[index, y1coord] = row['y2']

    # remove nearby horizontal lines
    horizontal1 = horizontal.copy()
    horizontal1 = horizontal1.sort_values('y1')
    horizontal1.index = range(len(horizontal1))
    horizontal1['ignore'] = 0
    horizontal1['lag1'] = horizontal1['y1'] - horizontal1['y1'].shift(1)
    horizontal1 = horizontal1.fillna(0.0)
    i = 1
    yccord = horizontal1.columns.get_loc('lag1')
    for index, row in horizontal1.iterrows():
        if (row['lag1'] < 25):
            horizontal1.iloc[index, yccord] = str(i) + 'a'
        else:
            i += 1  # save this i for further processing

    # avoid unnecessary joinings
    for j in range(i):
        temp1 = horizontal1[horizontal1['lag1'] == str(j + 1) + 'a']
        if (temp1.empty):
            continue
        if j > 0:
            temp1 = pd.concat([pd.DataFrame(horizontal1.iloc[temp1.index[0] - 1]).transpose(), temp1])
            for i in temp1.index:
                horizontal1.iloc[i, horizontal1.columns.get_loc('ignore')] = 1
            yvalue = temp1[temp1['xdiff'] == temp1.xdiff.max()].y1.copy()
            yvalue = yvalue[yvalue.index[0]]
            df = pd.DataFrame(
                [[temp1.x1.min(), yvalue, temp1.x2.max(), yvalue, abs(temp1.x1.min() - temp1.x2.max()), 0, 0]],
                columns=horizontal1.columns, index=[len(horizontal1)])
            horizontal1 = pd.concat([horizontal1, df])
        else:
            for i in temp1.index:
                horizontal1.iloc[i, horizontal1.columns.get_loc('ignore')] = 1
            yvalue = temp1[temp1['xdiff'] == temp1.xdiff.max()].y1
            yvalue = yvalue[yvalue.index[0]]
            df = pd.DataFrame(
                [[temp1.x1.min(), yvalue, temp1.x2.max(), yvalue, abs(temp1.x1.min() - temp1.x2.max()), 0, 0]],
                columns=horizontal1.columns, index=[len(horizontal1)])
            horizontal1 = pd.concat([horizontal1, df])

    # remove nearby vertical lines
    vertical1 = vertical.copy()
    vertical1 = vertical1.sort_values('x1')
    vertical1.index = range(len(vertical1))
    vertical1['ignore'] = 0
    vertical1['lag1'] = vertical1['x1'] - vertical1['x1'].shift(1)
    vertical1 = vertical1.fillna(0.0)
    i = 1
    xccord = vertical1.columns.get_loc('lag1')
    for index, row in vertical1.iterrows():
        if (row['lag1'] < 25):
            vertical1.iloc[index, xccord] = str(i) + 'a'
        else:
            i += 1  # save this i for further processing
    for j in range(i):
        try:
            temp1 = vertical1[vertical1['lag1'] == str(j + 1) + 'a']
        except:
            exit()
        if (temp1.empty):
            continue
        if j > 0:
            temp1 = pd.concat([pd.DataFrame(vertical1.iloc[temp1.index[0] - 1]).transpose(), temp1])
            for i in temp1.index:
                vertical1.iloc[i, vertical1.columns.get_loc('ignore')] = 1
            xvalue = temp1[temp1['ydiff'] == temp1.ydiff.max()].x1

            xvalue = xvalue[xvalue.index[0]]
            df = pd.DataFrame(
                [[xvalue, temp1.y2.max(), xvalue, temp1.y1.min(), abs(temp1.y1.min() - temp1.y2.max()), 0, 0]],
                columns=vertical1.columns, index=[len(vertical1)])
            vertical1 = pd.concat([vertical1, df])
        else:
            for i in temp1.index:
                vertical1.iloc[i, vertical1.columns.get_loc('ignore')] = 1
            xvalue = temp1[temp1['ydiff'] == temp1.ydiff.max()].x1

            xvalue = xvalue[xvalue.index[0]]
            df = pd.DataFrame(
                [[xvalue, temp1.y2.max(), xvalue, temp1.y1.min(), abs(temp1.y1.min() - temp1.y2.max()), 0, 0]],
                columns=vertical1.columns, index=[len(vertical1)])
            vertical1 = pd.concat([vertical1, df])

    # pre processing

    # Make x1 < x2 for horizontal lines
    horizontal2 = horizontal1.copy()
    horizontal2 = horizontal2[horizontal2['ignore'] == 0]
    horizontal2.index = range(len(horizontal2))

    vertical2 = vertical1.copy()
    vertical2 = vertical2[vertical2['ignore'] == 0]
    vertical2.index = range(len(vertical2))
    # Make x1 < x2 for horizontal lines

    temp = horizontal2[horizontal2.x2 < horizontal2.x1]
    if len(temp) > 0:
        x1coord = horizontal2.columns.get_loc('x1')
        x2coord = horizontal2.columns.get_loc('x2')
        for index, row in temp.iterrows():
            horizontal2.iloc[index, x1coord] = row['x2']
            horizontal2.iloc[index, x2coord] = row['x1']
    # Make y1 < y2 for vertical lines
    temp = vertical2[vertical2.y2 < vertical2.y1]
    if len(temp) > 0:
        y1coord = vertical2.columns.get_loc('y1')
        y2coord = vertical2.columns.get_loc('y2')
        for index, row in temp.iterrows():
            vertical2.iloc[index, y1coord] = row['y2']
            vertical2.iloc[index, y2coord] = row['y1']

    #######################################################################
    # remove the horizontal line segments which are not intersecting at all#
    #######################################################################

    deleterows = []
    for index, hrow in horizontal2.iterrows():
        intersected = False
        for _, vrow in vertical2.iterrows():
            result = lineprop.intersection([hrow.x1 - 20, hrow.y1, hrow.x2 + 20, hrow.y2],
                                   [vrow.x1, vrow.y1 - 50, vrow.x2, vrow.y2 + 50])
            if result[0]:
                intersected = True
                break
        if not intersected:
            deleterows.append(index)
    if len(deleterows) > 0:
        horizontal2 = horizontal2.drop(horizontal2.index[deleterows])
        horizontal2.index = range(len(horizontal2))

    # line extrapolation till the maximum allowed level
    # extend horizontal lines
    vertical2 = vertical1[vertical1['ignore'] == 0]
    vertical2.index = range(len(vertical2))

    # Make y1 < y2 for vertical2 lines
    temp = vertical2[vertical2.y2 < vertical2.y1]
    if len(temp) > 0:
        y1coord = vertical2.columns.get_loc('y1')
        y2coord = vertical2.columns.get_loc('y2')
        for index, row in temp.iterrows():
            vertical2.iloc[index, y1coord] = row['y2']
            vertical2.iloc[index, y2coord] = row['y1']

    x1coord = horizontal2.columns.get_loc('x1')
    x2coord = horizontal2.columns.get_loc('x2')
    for index, row in horizontal2.iterrows():
        # extend left
        temp = vertical2[(vertical2.y1 - 50 <= row.y1) & (vertical2.y2 + 50 >= row.y1) & (vertical2.x1 < row.x1 + 50)]
        if (len(temp) > 0):
            horizontal2.iloc[index, x1coord] = temp.x1.min()
        # extend right
        temp = vertical2[(vertical2.y1 - 50 <= row.y1) & (vertical2.y2 + 50 >= row.y1) & (vertical2.x2 > row.x2 - 50)]
        if (len(temp) > 0):
            horizontal2.iloc[index, x2coord] = temp.x2.max()

    # extend vertical lines (change logic -- to the nearest horizontal line) ************
    y1coord = vertical2.columns.get_loc('y1')
    y2coord = vertical2.columns.get_loc('y2')
    for index, row in vertical2.iterrows():
        # extend top
        temp = horizontal2[
            (horizontal2.x1 - 50 <= row.x1) & (horizontal2.x2 + 50 >= row.x1) & (horizontal2.y1 <= row.y1)]
        if (len(temp) > 0):
            vertical2.iloc[index, y1coord] = temp.y1.min()
        # extend bottom
        temp = horizontal2[
            (horizontal2.x1 - 50 <= row.x1) & (horizontal2.x2 + 50 >= row.x1) & (horizontal2.y2 >= row.y2)]
        if (len(temp) > 0):
            vertical2.iloc[index, y2coord] = temp.y2.max()

    horizontal2 = horizontal2.sort_values(['y1'])
    vertical2 = vertical2.sort_values(['x1'])

    vertixdf = pd.DataFrame(columns=['x1', 'y1', 'hindex', 'yindex'])
    for index, hrow in horizontal2.iterrows():
        for index1, vrow in vertical2.iterrows():
            result = lineprop.intersection([hrow.x1, hrow.y1, hrow.x2, hrow.y2], [vrow.x1, vrow.y1, vrow.x2, vrow.y2])
            if result[0]:
                df = pd.DataFrame([[result[1][0], result[1][1], index, index1]], columns=vertixdf.columns,
                                  index=[len(vertixdf)])
                vertixdf = pd.concat([vertixdf, df])

    # Prepare bounding boxes
    ########################
    boxlist = get_boxes_using_vertices(horizontal, vertical, vertixdf)

    """clr = (255, 0 , 0)
    for box in boxlist:
        a, b, c, d = box
        th = 10
        cv2.line(img_to_write_cpy, (a[0]+th,a[1]+th), (b[0]-th,b[1]+th), clr,3)
        cv2.line(img_to_write_cpy, (b[0]-th,b[1]+th), (c[0]-th,c[1]-th), clr,3)
        cv2.line(img_to_write_cpy, (c[0]-th,c[1]-th), (d[0]+th,d[1]-th), clr,3)
        cv2.line(img_to_write_cpy, (d[0]+th,d[1]-th), (a[0]+th,a[1]+th), clr,3)     
    cv2.imwrite(output_image_name + "all.png", img_to_write_cpy)"""
    # print("#############$$$$$$$$##############")
    # print(boxlist)
    # print("#############%%%%%%%%##############")
    ###################################### Leptonica cleansing code starts ####################################
    if len(leptonica_boxes) > 0:
        if len(boxlist) > 1:
            changed_vertix, vertixdf = fix_vertix_due_to_hor_junk(boxlist, vertixdf, leptonica_boxes)
            if changed_vertix:
                boxlist = get_boxes_using_vertices(horizontal, vertical, vertixdf)

        """changed_vertix, vertixdf = fix_vertix_due_to_vert_junk(boxlist, vertixdf, leptonica_boxes)
        if changed_vertix:
            boxlist = get_boxes_using_vertices(horizontal,vertical,vertixdf)"""
    ###################################### Leptonica cleansing code ends  ####################################

    ###########################
    # find tables using boxes
    ###########################
    from operator import itemgetter

    boxlist = sorted(boxlist, key=itemgetter(1, 0))
    box_dict = {
        "x": [],
        "y": [],
        "box": []
    }
    for box in boxlist:
        box_dict['x'].append(box[0][0])
        box_dict['y'].append(box[0][1])
        box_dict['box'].append(box)

    box_df = pd.DataFrame(box_dict)
    box_df = box_df.sort_values(['y', 'x'])
    box_df.index = range(len(box_df))
    sorted_boxlist = list(box_df.box)
    boxlist = sorted_boxlist
    revc_sorted_boxlist = list(box_df.sort_values(['y', 'x'], ascending=False)['box'])

    usedboxes = []
    tableslist = []
    table = []
    while (len(usedboxes) != len(boxlist)):
        for box in boxlist:
            if box in usedboxes:
                continue
            if len(table) == 0:  # add the first box
                table.append(box)
                usedboxes.append(box)
            for tbox in table:
                if ((box[0] == tbox[1]) & (box[3] == tbox[2])):  # adjacent box? sharing the line? ##### add treshold
                    table.append(box)
                    usedboxes.append(box)
                    break
                if ((box[0][1] == tbox[0][1]) & (
                        box[0] == tbox[1])):  # adjacent box? sharing the line? ##### add treshold
                    table.append(box)
                    usedboxes.append(box)
                    break
                if ((box[0] == tbox[3]) or (box[1] == tbox[2])):  # downside box? sharing the line?
                    table.append(box)
                    usedboxes.append(box)
                    break

        for box in revc_sorted_boxlist:
            if box in usedboxes:
                continue
            for tbox in table:
                if (
                        (tbox[0] == box[
                            1])):  # & (box[3] == tbox[2])):  # adjacent box? sharing the line? ##### add treshold
                    table.append(box)
                    usedboxes.append(box)
                    break

                if ((tbox[0] == box[3]) or (tbox[1] == box[2])):  # up side box? sharing the line?
                    table.append(box)
                    usedboxes.append(box)
                    break

        if (len(table)):
            tableslist.append(table)
            table = []

    return tableslist, boxlist, img_to_write_cpy


def get_boxes_using_vertices(horizontal, vertical, vertixdf):
    boxlist = []
    # for index, row in vertixdf.iterrows():
    for index, row in vertixdf.sort_index().iterrows():
        topleft = (row.x1, row.y1)
        ### find all the right side points on the same line segment
        rightx = vertixdf[(vertixdf.y1 == row.y1) & (vertixdf.x1 > row.x1) & (vertixdf.hindex == row.hindex)]
        if rightx.empty:
            continue
        rightx = rightx.sort_values(['x1'])
        rightx.index = range(len(rightx))

        topright = ""
        for index1, row1 in rightx.iterrows():
            topright = (row1.x1, row.y1)
            ### Ensure topleft and topright are connected horizontally
            ### also include topright to topleft with and below done
            if ((horizontal[((horizontal.y1 >= row.y1 - 10) & (horizontal.y1 <= row.y1 + 10)) & \
                            ((horizontal.x1 <= row.x1 + 10) & (horizontal.x2 > row.x1 + 10))].empty)):
                topright = ""
                break
            ### there exists a linesegment (vertical) downwords rightx.x1, row.y1?
            if (vertical[((vertical.x1 >= row1.x1 - 10) & (vertical.x1 <= row1.x1 + 10)) & \
                         ((vertical.y1 <= row.y1 + 10) & (vertical.y2 > row.y1 + 10))].empty):
                topright = ""
                continue
            else:
                break
        if topright == "":
            continue
        ### find all the right side points on the same line segment
        bottomy = vertixdf[(vertixdf.x1 == row.x1) & (vertixdf.y1 > row.y1) & (vertixdf.yindex == row.yindex)]
        if bottomy.empty:
            continue
        bottomy = bottomy.sort_values(['y1'])
        bottomy.index = range(len(bottomy))
        bottomleft = ""
        for _, row2 in bottomy.iterrows():
            bottomleft = (row.x1, row2.y1)

            ### Ensure topleft and bottomleft are connected vertically or bottomleft to topleft
            if ((vertical[((vertical.x1 >= row.x1 - 10) & (vertical.x1 <= row.x1 + 10)) & \
                          ((vertical.y1 <= row.y1 + 10) & (vertical.y2 > row.y1 + 10))].empty)):
                bottomleft = ""
                break
            ###there exists a linesegment rightside of  row.x1, bottomy.y1?
            if (horizontal[((horizontal.y1 >= row2.y1 - 10) & (horizontal.y1 <= row2.y1 + 10)) & \
                           ((horizontal.x1 <= row.x1 + 10) & (horizontal.x2 > row.x1 + 10))].empty):
                bottomleft = ""
                continue
            else:
                break
        if bottomleft == "":
            continue
        bottomright = (topright[0], bottomleft[1])
        boxlist.append([topleft, topright, bottomright, bottomleft])
    return boxlist


'''
def _generate_json(lines):
    lines_df = pd.DataFrame(lines, columns=['x1', 'y1', 'x2', 'y2'])
    # _, css_string, json_tab_string = create_table_json(lines_df)
    # json_t = json.loads(str(json_tab_string).replace('\'', '"'))
    # return json_t, css_string

'''



###############################################################
###              Leptonica cleancing code                   ###
###############################################################

def get_leptonica_boxes(input_dir, leptonica_dir, file):

    # PROJECT = cfg['project']
    # SUBPROJECT = cfg['subproject']

    # input_dir = cfg[PROJECT][SUBPROJECT]['leptonica-dir']
    # output_dir = cfg[PROJECT][SUBPROJECT]['output-dir']


    file_name = '.'.join(file.split('.')[:-1])

    # Get the image
    img = image_loader.get_image(input_dir, file)
    char_img = image_loader.get_image(leptonica_dir, file)
    img = remove_text(char_img,img)

    # Clean the image by removing shades
    clean_image = image_cleaner.remove_shades(img, True, file_name)
    # Dotted lines detection post OCR
    # Split the image into seprate table images
    # table_rois, bboxline_coords = table_image_splitter.split_image_2_table_images(clean_image, file_name)
    table_rois, bboxline_coords, bboxs = table_image_splitter.split_image_2_table_images(clean_image,40,75,'all',
                                                                                         file_name)
    img_to_write = img
    rois_len = len(table_rois)
    rois_counter = 0
    debug_img = img.copy()
    all_boxes = []
    for img_snip, bbox_lines in zip(table_rois, bboxline_coords):
        snip = img_snip.copy()
        snip = snip.astype('uint8')
        rois_counter += 1
        if (rois_counter == rois_len):
            echo_flag = True
        else:
            echo_flag = False
        boxlist = detect_leptonica_table(snip, bbox_lines, img_to_write, img)
        all_boxes += boxlist

    clr = (0, 0, 255)
    for box in all_boxes:
        a, b, c, d = box
        th = 10

        cv2.line(debug_img, (int(a[0]) + th, int(a[1]) + th), (int(b[0]) - th, int(b[1]) + th), clr, 3)
        cv2.line(debug_img, (int(b[0]) - th, int(b[1]) + th), (int(c[0]) - th, int(c[1]) - th), clr, 3)
        cv2.line(debug_img, (int(c[0]) - th, int(c[1]) - th), (int(d[0]) + th, int(d[1]) - th), clr, 3)
        cv2.line(debug_img, (int(d[0]) + th, int(d[1]) - th), (int(a[0]) + th, int(a[1]) + th), clr, 3)
    cv2.imwrite(output_dir + file_name + "lept.png", debug_img)

    return all_boxes


def detect_leptonica_table(snip, bbox_lines, img_to_write, orig_img):
    roi_lines = line_detection.detect_lines(snip, bbox_lines, orig_img)
    roi_points, roi_lines = point_detection.detect_points(roi_lines)
    boxlist = create_bound_boxs_from_leptonica_lines(roi_lines)
    return boxlist


def remove_text(rgb_char, rgb_line):
    hasText = 0
    gray = cv2.cvtColor(rgb_char, cv2.COLOR_BGR2GRAY);
    morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, morphKernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # find contours
    mask = np.zeros(bw.shape[:2], dtype="uint8")
    _, contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours
    idx = 0
    widths = []
    heights = []
    while idx >= 0:
        x, y, w, h = cv2.boundingRect(contours[idx])
        widths.append(w)
        heights.append(w)
        idx = hierarchy[0][idx][0]

    min_width = np.mean(widths) / 2
    max_width = np.mean(widths) + min_width
    min_height = np.mean(widths) / 2
    max_height = np.mean(widths) + min_width

    idx = 0
    while idx >= 0:
        x, y, w, h = cv2.boundingRect(contours[idx]);
        # ratio of non-zero pixels in the filled region
        r = cv2.contourArea(contours[idx]) / (w * h)
        if ((h < 7 or w < 7)):
            pass
        else:
            cv2.drawContours(rgb_line, contours, idx, (255, 255, 255), cv2.FILLED)
        idx = hierarchy[0][idx][0]
    return rgb_line

def create_bound_boxs_from_leptonica_lines(lines):
    lines_df = pd.DataFrame(lines, columns=['x1', 'y1', 'x2', 'y2'])

    lines_df['xdiff'] = abs(lines_df['x1'] - lines_df['x2'])
    lines_df['ydiff'] = abs(lines_df['y1'] - lines_df['y2'])

    ###find those lines which are neither parallel to x-axis nor y-axis and fix them
    tempdata = lines_df[(lines_df['x1'] != lines_df['x2']) & (lines_df['y1'] != lines_df['y2'])]
    for index, row in tempdata.iterrows():
        if (row['xdiff'] > 2 and row['ydiff'] > 2):
            continue
        if (row['xdiff'] < row['ydiff']):
            lines_df.iloc[index, lines_df.columns.get_loc('x2')] = row['x1']
        else:
            lines_df.iloc[index, lines_df.columns.get_loc('y2')] = row['y1']

    # split the data into vertical and horizontal lines
    horizontal = lines_df[lines_df['y1'] == lines_df['y2']]
    horizontal = horizontal.drop(['ydiff'], axis=1)
    horizontal = horizontal.sort_values('xdiff')
    horizontal.index = range(len(horizontal))
    vertical = lines_df[lines_df['x1'] == lines_df['x2']]
    vertical = vertical.drop(['xdiff'], axis=1)
    vertical = vertical.sort_values('ydiff')
    vertical.index = range(len(vertical))

    # Make x1 < x2 for vertical lines
    temp = horizontal[horizontal.x2 < horizontal.x1]
    if len(temp) > 0:
        x1coord = horizontal.columns.get_loc('x1')
        x2coord = horizontal.columns.get_loc('x2')
        for index, row in temp.iterrows():
            horizontal.iloc[index, x1coord] = row['x2']
            horizontal.iloc[index, x2coord] = row['x1']
    # Make y1 < y2 for vertical lines
    temp = vertical[vertical.y2 < vertical.y1]
    if len(temp) > 0:
        y1coord = vertical.columns.get_loc('y1')
        y2coord = vertical.columns.get_loc('y2')
        for index, row in temp.iterrows():
            vertical.iloc[index, y1coord] = row['y2']

    # remove nearby horizontal lines
    horizontal1 = horizontal.copy()
    horizontal1 = horizontal1.sort_values('y1')
    horizontal1.index = range(len(horizontal1))
    horizontal1['ignore'] = 0
    horizontal1['lag1'] = horizontal1['y1'] - horizontal1['y1'].shift(1)
    horizontal1 = horizontal1.fillna(0.0)
    i = 1
    yccord = horizontal1.columns.get_loc('lag1')
    for index, row in horizontal1.iterrows():
        if (row['lag1'] < 25):
            horizontal1.iloc[index, yccord] = str(i) + 'a'
        else:
            i += 1  # save this i for further processing

    # avoid unnecessary joinings
    for j in range(i):
        temp1 = horizontal1[horizontal1['lag1'] == str(j + 1) + 'a']
        if (temp1.empty):
            continue
        if j > 0:
            temp1 = pd.concat([pd.DataFrame(horizontal1.iloc[temp1.index[0] - 1]).transpose(), temp1])
            for i in temp1.index:
                horizontal1.iloc[i, horizontal1.columns.get_loc('ignore')] = 1
            yvalue = temp1[temp1['xdiff'] == temp1.xdiff.max()].y1.copy()
            yvalue = yvalue[yvalue.index[0]]
            df = pd.DataFrame(
                [[temp1.x1.min(), yvalue, temp1.x2.max(), yvalue, abs(temp1.x1.min() - temp1.x2.max()), 0, 0]],
                columns=horizontal1.columns, index=[len(horizontal1)])
            horizontal1 = pd.concat([horizontal1, df])
        else:
            for i in temp1.index:
                horizontal1.iloc[i, horizontal1.columns.get_loc('ignore')] = 1
            yvalue = temp1[temp1['xdiff'] == temp1.xdiff.max()].y1
            yvalue = yvalue[yvalue.index[0]]
            df = pd.DataFrame(
                [[temp1.x1.min(), yvalue, temp1.x2.max(), yvalue, abs(temp1.x1.min() - temp1.x2.max()), 0, 0]],
                columns=horizontal1.columns, index=[len(horizontal1)])
            horizontal1 = pd.concat([horizontal1, df])

    # remove nearby vertical lines
    vertical1 = vertical.copy()
    vertical1 = vertical1.sort_values('x1')
    vertical1.index = range(len(vertical1))
    vertical1['ignore'] = 0
    vertical1['lag1'] = vertical1['x1'] - vertical1['x1'].shift(1)
    vertical1 = vertical1.fillna(0.0)
    i = 1
    xccord = vertical1.columns.get_loc('lag1')
    for index, row in vertical1.iterrows():
        if (row['lag1'] < 25):
            vertical1.iloc[index, xccord] = str(i) + 'a'
        else:
            i += 1  # save this i for further processing
    for j in range(i):
        try:
            temp1 = vertical1[vertical1['lag1'] == str(j + 1) + 'a']
        except:
            exit()
        if (temp1.empty):
            continue
        if j > 0:
            temp1 = pd.concat([pd.DataFrame(vertical1.iloc[temp1.index[0] - 1]).transpose(), temp1])
            for i in temp1.index:
                vertical1.iloc[i, vertical1.columns.get_loc('ignore')] = 1
            xvalue = temp1[temp1['ydiff'] == temp1.ydiff.max()].x1

            xvalue = xvalue[xvalue.index[0]]
            df = pd.DataFrame(
                [[xvalue, temp1.y2.max(), xvalue, temp1.y1.min(), abs(temp1.y1.min() - temp1.y2.max()), 0, 0]],
                columns=vertical1.columns, index=[len(vertical1)])
            vertical1 = pd.concat([vertical1, df])
        else:
            for i in temp1.index:
                vertical1.iloc[i, vertical1.columns.get_loc('ignore')] = 1
            xvalue = temp1[temp1['ydiff'] == temp1.ydiff.max()].x1

            xvalue = xvalue[xvalue.index[0]]
            df = pd.DataFrame(
                [[xvalue, temp1.y2.max(), xvalue, temp1.y1.min(), abs(temp1.y1.min() - temp1.y2.max()), 0, 0]],
                columns=vertical1.columns, index=[len(vertical1)])
            vertical1 = pd.concat([vertical1, df])

    # pre processing

    # Make x1 < x2 for horizontal lines
    horizontal2 = horizontal1.copy()
    horizontal2 = horizontal2[horizontal2['ignore'] == 0]
    horizontal2.index = range(len(horizontal2))

    vertical2 = vertical1.copy()
    vertical2 = vertical2[vertical2['ignore'] == 0]
    vertical2.index = range(len(vertical2))
    # Make x1 < x2 for horizontal lines

    temp = horizontal2[horizontal2.x2 < horizontal2.x1]
    if len(temp) > 0:
        x1coord = horizontal2.columns.get_loc('x1')
        x2coord = horizontal2.columns.get_loc('x2')
        for index, row in temp.iterrows():
            horizontal2.iloc[index, x1coord] = row['x2']
            horizontal2.iloc[index, x2coord] = row['x1']
    # Make y1 < y2 for vertical lines
    temp = vertical2[vertical2.y2 < vertical2.y1]
    if len(temp) > 0:
        y1coord = vertical2.columns.get_loc('y1')
        y2coord = vertical2.columns.get_loc('y2')
        for index, row in temp.iterrows():
            vertical2.iloc[index, y1coord] = row['y2']
            vertical2.iloc[index, y2coord] = row['y1']

    #######################################################################
    # remove the horizontal line segments which are not intersecting at all#
    #######################################################################

    deleterows = []
    for index, hrow in horizontal2.iterrows():
        intersected = False
        for _, vrow in vertical2.iterrows():
            result = lineprop.intersection([hrow.x1 - 20, hrow.y1, hrow.x2 + 20, hrow.y2],
                                   [vrow.x1, vrow.y1 - 50, vrow.x2, vrow.y2 + 50])
            if result[0]:
                intersected = True
                break
        if not intersected:
            deleterows.append(index)
    if len(deleterows) > 0:
        horizontal2 = horizontal2.drop(horizontal2.index[deleterows])
        horizontal2.index = range(len(horizontal2))

    # line extrapolation till the maximum allowed level
    # extend horizontal lines
    vertical2 = vertical1[vertical1['ignore'] == 0]
    vertical2.index = range(len(vertical2))

    # Make y1 < y2 for vertical2 lines
    temp = vertical2[vertical2.y2 < vertical2.y1]
    if len(temp) > 0:
        y1coord = vertical2.columns.get_loc('y1')
        y2coord = vertical2.columns.get_loc('y2')
        for index, row in temp.iterrows():
            vertical2.iloc[index, y1coord] = row['y2']
            vertical2.iloc[index, y2coord] = row['y1']

    x1coord = horizontal2.columns.get_loc('x1')
    x2coord = horizontal2.columns.get_loc('x2')
    for index, row in horizontal2.iterrows():
        # extend left
        temp = vertical2[(vertical2.y1 - 50 <= row.y1) & (vertical2.y2 + 50 >= row.y1) & (vertical2.x1 < row.x1 + 50)]
        if (len(temp) > 0):
            horizontal2.iloc[index, x1coord] = temp.x1.max()
        # extend right
        temp = vertical2[(vertical2.y1 - 50 <= row.y1) & (vertical2.y2 + 50 >= row.y1) & (vertical2.x2 > row.x2 - 50)]
        if (len(temp) > 0):
            horizontal2.iloc[index, x2coord] = temp.x2.min()

    # extend vertical lines (change logic -- to the nearest horizontal line) ************
    y1coord = vertical2.columns.get_loc('y1')
    y2coord = vertical2.columns.get_loc('y2')
    for index, row in vertical2.iterrows():
        # extend top
        temp = horizontal2[
            (horizontal2.x1 - 50 <= row.x1) & (horizontal2.x2 + 50 >= row.x1) & (horizontal2.y1 <= row.y1)]
        if (len(temp) > 0):
            vertical2.iloc[index, y1coord] = temp.y1.max()
        # extend bottom
        temp = horizontal2[
            (horizontal2.x1 - 50 <= row.x1) & (horizontal2.x2 + 50 >= row.x1) & (horizontal2.y2 >= row.y2)]
        if (len(temp) > 0):
            vertical2.iloc[index, y2coord] = temp.y2.min()

    horizontal2 = horizontal2.sort_values(['y1'])
    vertical2 = vertical2.sort_values(['x1'])

    vertixdf = pd.DataFrame(columns=['x1', 'y1', 'hindex', 'yindex'])
    for index, hrow in horizontal2.iterrows():
        for index1, vrow in vertical2.iterrows():
            result = lineprop.intersection([hrow.x1, hrow.y1, hrow.x2, hrow.y2], [vrow.x1, vrow.y1, vrow.x2, vrow.y2])
            if result[0]:
                df = pd.DataFrame([[result[1][0], result[1][1], index, index1]], columns=vertixdf.columns,
                                  index=[len(vertixdf)])
                vertixdf = pd.concat([vertixdf, df])

    boxlist = get_boxes_using_vertices(horizontal, vertical, vertixdf)
    return boxlist


def fix_vertix_due_to_vert_junk(boxlist, vertixdf, lept_boxes):
    import math
    import pandas as pd
    import numpy as np

    lept_lines_df = get_lept_lines_df(lept_boxes)
    line1_dict = {
        "tx": [],
        "ty": [],
        "bx": [],
        "by": [],
        "ydiff": [],
        "xdiff": [],
        "box": []
    }
    for box in boxlist:
        a, b, c, d = box
        line1_dict['tx'].append(a[0])
        line1_dict['ty'].append(a[1])
        line1_dict['bx'].append(c[0])
        line1_dict['by'].append(c[1])
        line1_dict['ydiff'].append(abs(a[1] - d[1]))
        line1_dict['xdiff'].append(abs(a[0] - b[0]))
        line1_dict['box'].append(box)

    line1df = pd.DataFrame(line1_dict)

    line1df = line1df.sort_values('tx')
    line1df = line1df.reset_index(drop=True)

    line1df_counts = line1df.groupby("tx").size().reset_index(name='counts')
    avg_cnt = math.ceil(line1df_counts.counts.mean())  # is this avg fine?
    line1df_counts['deviation_perc'] = line1df_counts['counts'].apply(lambda x: x / avg_cnt)
    line1df = pd.merge(line1df, line1df_counts, on=['tx'])
    xdiff_mode_df = line1df.groupby(["bx"]).agg(lambda x: x.value_counts().index[0]).reset_index()[['bx', 'xdiff']]
    xdiff_mode_df.columns = ['bx', 'xdiff_mode']
    line1df = pd.merge(line1df, xdiff_mode_df, on=['bx'])

    line1df['leftIssue'] = 0
    line1df.loc[(line1df["deviation_perc"] < 0.5) & (line1df.xdiff < line1df.xdiff_mode), 'leftIssue'] = 1

    line1df = line1df[['tx', 'ty', 'bx', 'by', 'ydiff', 'xdiff', 'box', 'leftIssue']]
    line1df = line1df.sort_values('bx')
    line1df = line1df.reset_index(drop=True)

    line1df_counts = line1df.groupby("bx").size().reset_index(name='counts')
    avg_cnt = math.floor(line1df_counts.counts.mean())  # is this avg fine?

    line1df_counts['deviation_perc'] = line1df_counts['counts'].apply(lambda x: x / avg_cnt)
    line1df = pd.merge(line1df, line1df_counts, on=['bx'])
    xdiff_mode_df = line1df.groupby(["tx"]).agg(lambda x: x.value_counts().index[0]).reset_index()[['tx', 'xdiff']]
    xdiff_mode_df.columns = ['tx', 'xdiff_mode']
    line1df = pd.merge(line1df, xdiff_mode_df, on=['tx'])
    line1df['rightIssue'] = 0
    line1df.loc[(line1df["deviation_perc"] < 0.5) & (line1df.xdiff < line1df.xdiff_mode), 'rightIssue'] = 1

    box_issue_df = line1df[(line1df.leftIssue == 1) | (line1df.rightIssue == 1)]

    if len(box_issue_df) > 0:
        vertixdf["ignore"] = 0
        for _, row in box_issue_df.iterrows():
            if int(row.leftIssue):
                # print("left", row.box)
                suspect_line = [row.box[0], row.box[-1]]
                if (find_lept_lines(suspect_line, lept_lines_df)):
                    continue
                for vertix in suspect_line:
                    tx, ty = vertix
                    vertixdf.loc[(vertixdf.x1 == tx) & (vertixdf.y1 == ty), "ignore"] = 1
            else:
                # print("right", row.box)
                suspect_line = row.box[1:3]
                if (find_lept_lines(suspect_line, lept_lines_df)):
                    continue
                for vertix in suspect_line:
                    tx, ty = vertix
                    vertixdf.loc[(vertixdf.x1 == tx) & (vertixdf.y1 == ty), "ignore"] = 1
        vertixdf = vertixdf[vertixdf.ignore != 1]
        return True, vertixdf[['x1', 'y1', 'hindex', 'yindex']]
    else:
        return False, vertixdf


def fix_vertix_due_to_hor_junk(boxlist, vertixdf, lept_boxes):
    import math
    import pandas as pd
    import numpy as np
    lept_lines_df = get_lept_lines_df(lept_boxes)
    line1_dict = {
        "tx": [],
        "ty": [],
        "bx": [],
        "by": [],
        "ydiff": [],
        "xdiff": [],
        "box": []
    }
    # Create dataframe with lines
    for box in boxlist:
        a, b, c, d = box
        line1_dict['tx'].append(a[0])
        line1_dict['ty'].append(a[1])
        line1_dict['bx'].append(c[0])
        line1_dict['by'].append(c[1])
        line1_dict['ydiff'].append(abs(a[1] - d[1]))
        line1_dict['xdiff'].append(abs(a[0] - b[0]))
        line1_dict['box'].append(box)

    line1df = pd.DataFrame(line1_dict)
    line1df = line1df.sort_values('ty')
    line1df = line1df.reset_index(drop=True)

    # find each top lines box count
    line1df_counts = line1df.groupby("ty").size().reset_index(name='counts')
    avg_cnt = math.floor(line1df_counts.counts.mean())  # is this avg fine?
    # deviation from each line's box count to all lines average box count
    line1df_counts['deviation_perc'] = line1df_counts['counts'].apply(lambda x: x / avg_cnt)
    line1df = pd.merge(line1df, line1df_counts, on=['ty'])
    # Mode of box's height sharing the same bottom line
    ydiff_mode_df = line1df.groupby(["by"]).agg(lambda x: x.value_counts().index[0]).reset_index()[['by', 'ydiff']]
    ydiff_mode_df.columns = ['by', 'ydiff_mode']
    line1df = pd.merge(line1df, ydiff_mode_df, on=['by'])
    line1df['topIssue'] = 0
    line1df.loc[(line1df["deviation_perc"] < 0.5) & (line1df.ydiff < line1df.ydiff_mode), 'topIssue'] = 1

    line1df = line1df[['tx', 'ty', 'bx', 'by', 'ydiff', 'xdiff', 'box', 'topIssue']]
    line1df = line1df.sort_values('by')
    line1df = line1df.reset_index(drop=True)

    # find each bottom lines box count
    line1df_counts = line1df.groupby("by").size().reset_index(name='counts')
    avg_cnt = math.floor(line1df_counts.counts.mean())  # is this avg fine?
    # deviation from each line's box count to all lines average box count
    line1df_counts['deviation_perc'] = line1df_counts['counts'].apply(lambda x: x / avg_cnt)
    line1df = pd.merge(line1df, line1df_counts, on=['by'])
    # Mode of box's height sharing the same bottom line
    ydiff_mode_df = line1df.groupby(["ty"]).agg(lambda x: x.value_counts().index[0]).reset_index()[['ty', 'ydiff']]
    ydiff_mode_df.columns = ['ty', 'ydiff_mode']
    line1df = pd.merge(line1df, ydiff_mode_df, on=['ty'])
    line1df['bottomIssue'] = 0
    line1df.loc[(line1df["deviation_perc"] < 0.5) & (line1df.ydiff < line1df.ydiff_mode), 'bottomIssue'] = 1
    box_issue_df = line1df[(line1df.topIssue == 1) | (line1df.bottomIssue == 1)]

    # don't delete if same kind of boxes not having any problems

    if len(box_issue_df) > 0:
        vertixdf["ignore"] = 0
        for _, row in box_issue_df.iterrows():
            if int(row.topIssue):
                suspect_line = row.box[:2]
                if (find_lept_lines(suspect_line, lept_lines_df)):
                    continue
                for vertix in suspect_line:
                    tx, ty = vertix
                    vertixdf.loc[(vertixdf.x1 == tx) & (vertixdf.y1 == ty), "ignore"] = 1
            else:
                suspect_line = row.box[2:]
                if (find_lept_lines(suspect_line, lept_lines_df)):
                    continue
                for vertix in suspect_line:
                    tx, ty = vertix
                    vertixdf.loc[(vertixdf.x1 == tx) & (vertixdf.y1 == ty), "ignore"] = 1
        vertixdf = vertixdf[vertixdf.ignore != 1]
        return True, vertixdf[['x1', 'y1', 'hindex', 'yindex']]
    else:
        return False, vertixdf


def get_lept_lines_df(lept_boxes):
    import pandas as pd
    lept_lines = []
    for box in lept_boxes:
        a, b, c, d = box
        lept_lines.append([a[0], a[1], b[0], b[1]])
        lept_lines.append([b[0], b[1], c[0], c[1]])
        lept_lines.append([d[0], d[1], c[0], c[1]])
        lept_lines.append([a[0], a[1], d[0], d[1]])
    lept_lines_df = pd.DataFrame(lept_lines, columns=['x1', 'y1', 'x2', 'y2'])

    # split the data into vertical and horizontal lines
    horizontal = lept_lines_df[lept_lines_df['y1'] == lept_lines_df['y2']]
    horizontal.index = range(len(horizontal))
    vertical = lept_lines_df[lept_lines_df['x1'] == lept_lines_df['x2']]
    vertical.index = range(len(vertical))

    # Make x1 < x2 for vertical lines
    temp = horizontal[horizontal.x2 < horizontal.x1]
    if len(temp) > 0:
        x1coord = horizontal.columns.get_loc('x1')
        x2coord = horizontal.columns.get_loc('x2')
        for index, row in temp.iterrows():
            horizontal.iloc[index, x1coord] = row['x2']
            horizontal.iloc[index, x2coord] = row['x1']
    # Make y1 < y2 for vertical lines
    temp = vertical[vertical.y2 < vertical.y1]
    if len(temp) > 0:
        y1coord = vertical.columns.get_loc('y1')
        y2coord = vertical.columns.get_loc('y2')
        for index, row in temp.iterrows():
            vertical.iloc[index, y1coord] = row['y2']

    lept_lines_df = pd.concat([horizontal, vertical], ignore_index=True)
    return lept_lines_df


def find_lept_lines(suspect_line, lept_lines_df):
    if lept_lines_df is None:
        return False
    point1, point2 = suspect_line
    x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
    if x1 > x2:  # swap
        x1 = x1 + x2
        x2 = x1 - x2
        x1 = x1 - x2
    if y1 > y2:  # swap
        y1 = y1 + y2
        y2 = y1 - y2
        y1 = y1 - y2
    found = False

    for _, row in lept_lines_df.iterrows():
        if ((abs(row.x1 - x1) < 15) & (abs(row.y1 - y1) < 15) & (abs(row.x2 - x2) < 15) & (abs(row.y2 - y2) < 15)):
            found = True
            break
    return found