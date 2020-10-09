import cv2
import numpy as np
from apporchid.common.config import cfg as cfg
import random
import apporchid.agora.cv.image.image_operations as imageop
from apporchid.common.logger import logger as logger
colours = [(0,255,255),(255,0,255),(255,255,0),(0,255,0),(0,0,255),(255,0,0)]
colours += [(0,127,127),(127,0,127),(127,127,0),(0,0,127),(127,0,0),(0,127,0),(127,255,127),(127,127,255)]

debug_dir = cfg["debug-dir"]
debug_flag = cfg['debug']


def split_image_2_table_images(img, hor_scale, vert_scale, contour_level, file_name='file1'):

    file_name = file_name + '.png'

    mask_dotted = imageop.detect_dotted_lines(img, file_name)
    gray_image = imageop.convert_to_gray(img)
    bw = cv2.adaptiveThreshold(~gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    if debug_flag:
        cv2.imwrite(debug_dir + file_name + '_shadow_input_' + contour_level + '.png', img)
        cv2.imwrite(debug_dir + file_name + '_gray_image_' + contour_level+ '.png', gray_image)
        cv2.imwrite(debug_dir + file_name + '_bw_' + contour_level+ '.png', bw)

    bboxs = get_all_rect_bbox(bw, hor_scale, vert_scale, contour_level, file_name)
    logger.debug('Contour Count before: '+str(len(bboxs)))

    # bboxs = remove_small_bboxs(bboxs)
    unique_table_bboxs, _ = eliminate_inner_rect_bbox_thresh(bboxs)
    if debug_flag:
        cnt = 0
        img_cpy = img.copy()
        for bbox in unique_table_bboxs:
            clr = colours[cnt % 14]
            x,y,w,h = bbox
            cv2.rectangle(img_cpy,(x,y), (x+w,y+h),clr,3)
            cnt += 1
        cv2.imwrite(debug_dir + file_name + '_preContours_' + contour_level + '.png', img_cpy)

    character_dim_cutoffs = []
    for box in unique_table_bboxs:
        x, y, w, h = box
        orig_img_snip = np.zeros(bw.shape, dtype = 'uint8')
        orig_img_snip[y:y + h + 1, x:x + w + 1] = bw[y:y + h + 1, x:x + w + 1]
        w_cutoff, h_cutoff = imageop.get_average_character_size(orig_img_snip)
        character_dim_cutoffs.append((w_cutoff, h_cutoff))

    rois, bbox, bboxline_coords = _find_rectangle_contours_and_clean2(bw, hor_scale, vert_scale, unique_table_bboxs,
                                                                      character_dim_cutoffs,mask_dotted, contour_level,
                                                                      file_name)

    logger.debug('Contour Count after: '+str(len(rois)))
    if debug_flag:
        cnt = 1
        for roi in rois:
            cv2.imwrite(debug_dir + file_name + '_postContour_'+ contour_level + str(cnt) +'.png', roi)
            cnt += 1

    if debug_flag:
        cnt = 0
        img_cpy = img.copy()
        for bb in bbox:
            clr = colours[cnt % 14]
            x,y,w,h = bb
            cv2.rectangle(img_cpy,(x,y), (x+w,y+h),clr,3)
            cnt += 1
        cv2.imwrite(debug_dir + file_name + '_postContours' + contour_level + '.png', img_cpy)


    return rois, bboxline_coords, bbox


def get_all_rect_bbox(bw, hor_scale, vert_scale, contour_level, file_name='file1'):

    horizontal = imageop.detect_horizontal(bw, scale=hor_scale)
    ## OLD
    vertical = imageop.detect_vertical(bw, scale=vert_scale)
    ##
    ## NEW
    # vertical = imageop.detect_vertical(bw, scale=100)
    ##
    mask = horizontal + vertical
    joints = cv2.bitwise_and(horizontal, vertical)
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    final_img, contours_init, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    joint_cutoff = 1
    if contour_level == 'small':
        contours = []
        for contour in contours_init:
            contours_poly = cv2.approxPolyDP(contour, 3, True)
            bound_rect = cv2.boundingRect(contours_poly)
            x, y, w, h = bound_rect
            if bool((w > 15) and (w < 100)) != bool((h > 15) and (h < 100)):
                contours.append(contour)
    if contour_level == 'big':
        contours = []
        for contour in contours_init:
            contours_poly = cv2.approxPolyDP(contour, 3, True)
            bound_rect = cv2.boundingRect(contours_poly)
            x, y, w, h = bound_rect
            if (w >= 100) and (h >= 100):
                contours.append(contour)
    if contour_level == 'all':
        contours = contours_init
        joint_cutoff = 2

    if debug_flag:
        cv2.imwrite(debug_dir + file_name + '_preContourMask_' +contour_level+ '.png', mask)

    bbox = []

    for contour in contours:

        contours_poly = cv2.approxPolyDP(contour, 3, True)
        bound_rect = cv2.boundingRect(contours_poly)

        x, y, w, h = bound_rect
        roi = joints[y:y + h, x:x + w]
        _, thresh1 = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        img_, joints_contours, __ = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(joints_contours) <= joint_cutoff:
            continue

        bbox.append((x, y, w, h))

    return bbox



def _find_rectangle_contours_and_clean(bw, hor_scale, vert_scale, bboxs, character_dim_cutoffs, file_name='file1'):

    rois = []
    bboxline_coords, bbox_out = [], []

    for bbox, char_dim_cutoff in zip(bboxs,character_dim_cutoffs):

        x,y,w,h = bbox
        snip_input = np.zeros(bw.shape, dtype = int)
        snip_input = snip_input.astype('uint8')
        snip_input[y:y+h+1, x:x+w+1] = bw[y:y+h+1, x:x+w+1]

        snip_input_copy = snip_input.copy()
        if debug_flag:
            cv2.imwrite(debug_dir + file_name + '_snip_input_bw' + '.png', snip_input_copy)

        horizontal = imageop.detect_horizontal(snip_input_copy, scale=hor_scale)
        ## OLD
        # vertical = imageop.detect_vertical_and_clean_text(snip_input_copy, char_dim_cutoff, scale=55)
        ##
        ## NEW
        vertical = imageop.detect_vertical_and_clean_text(snip_input_copy, char_dim_cutoff, scale=vert_scale)
        ##
        mask = horizontal + vertical
        joints = cv2.bitwise_and(horizontal, vertical)
        _, thresh = cv2.threshold(mask, 127, 255, 0)
        final_img, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        thresh_cpy = thresh.copy()

        if debug_flag:
            cv2.imwrite(debug_dir + file_name + '_maskHor' +str(random.randint(1,101))+ '.png', horizontal)
            cv2.imwrite(debug_dir + file_name + '_maskVert' + str(random.randint(1,101))+'.png', vertical)
            cv2.imwrite(debug_dir + file_name + '_mask' + str(random.randint(1,101))+'.png', mask)
            cv2.imwrite(debug_dir + file_name + '_joints' + '.png', joints)
            cv2.imwrite(debug_dir + file_name + '_thresh' + '.png', thresh)
            cv2.imwrite(debug_dir + file_name + '_find_contours' + '.png', final_img)
            cv2.imwrite(debug_dir + file_name + '_thrsh_cpy' + '.png', thresh_cpy)

        if debug_flag:
            debug_contour_counter = 0

        for contour in contours:

            all_black = np.zeros((bw.shape[0], bw.shape[1]), dtype=int)
            contours_poly = cv2.approxPolyDP(contour, 3, True)
            bound_rect = cv2.boundingRect(contours_poly)

            x, y, w, h = bound_rect
            roi = joints[y:y + h, x:x + w]
            if debug_flag:
                cv2.rectangle(snip_input_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
                row, col = np.where(joints)
                row, col = list(row), list(col)
                ind = list(zip(row, col))
                for j in ind:
                    cx, cy = j
                    cv2.circle(snip_input_copy, (int(cy), int(cx)), 5, (0, 0, 0), -5)

            if debug_flag:
                cv2.imwrite(debug_dir + file_name + '_roi' + str(debug_contour_counter) + '.png', roi)
                debug_contour_counter += 1

            _, thresh1 = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            img_, joints_contours, __ = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if debug_flag:
                cv2.imwrite(debug_dir + file_name + '_find_contours_joints' + str(debug_contour_counter) + '.png', img_)

            # if len(joints_contours) <= 2:
            #     continue

            all_black[y:y + h, x:x + w] = thresh_cpy[y:y + h, x:x + w]

            bbox_out.append((x, y, w, h))
            rois.append(all_black.copy())
            bboxline_coords.append([(x, y, x + w, y),
                                (x + w, y, x + w, y + h),
                                (x, y + h, x + w, y + h),
                                (x, y, x, y + h)])

    return rois, bbox_out, bboxline_coords


def _find_rectangle_contours_and_clean2(bw, hor_scale, vert_scale, bboxs, character_dim_cutoffs, mask_dotted,
                                        contour_level,file_name='file1'):
    if contour_level == 'all':
        vert_scale = 55
        # vert_scale -= 20
    overlap_areas, overlap_thresh_flags, overlap_bound_rects, rect_areas_img = get_overlap_area_tables(bboxs)
    counter = 0
    tot_unique_bboxs, tot_unique_rois, tot_unique_bboxline_coords = [],[],[]
    net_img = np.zeros(bw.shape, dtype=int)
    for bbox, char_dim_cutoff in zip(bboxs, character_dim_cutoffs):

        rois, bboxline_coords, bbox_out = [], [], []
        x, y, w, h = bbox

        single_table_snip = np.zeros(bw.shape, dtype=int)
        single_table_snip[y:y + h + 1, x:x + w + 1] = bw[y:y + h + 1, x:x + w + 1]
        overlap_areas_row = overlap_areas[counter]


        nonzero_cols = np.nonzero(overlap_areas_row)
        nonzero_cols = list(nonzero_cols)
        nonzero_cols = nonzero_cols[0]
        nonzero_cols = nonzero_cols.tolist()

        if len(nonzero_cols) > 0:
            rect_areas_ind = rect_areas_img[nonzero_cols+[counter]]
            max_rect_area_ind = np.max(rect_areas_ind)
            current_rect_area_ind = rect_areas_img[counter]

            if current_rect_area_ind == max_rect_area_ind:
                for col in nonzero_cols:
                    ltop_x, ltop_y, width, height = overlap_bound_rects[counter, col]
                    single_table_snip[ltop_y:ltop_y + height + 1, ltop_x:ltop_x + width + 1] = 0

        single_table_snip = single_table_snip.astype('uint8')
        snip_input_copy = single_table_snip.copy()


        if debug_flag:
            cv2.imwrite(debug_dir + file_name + '_snip_input_bw_'+contour_level+ str(random.randint(1, 101)) + '.png', snip_input_copy)

        horizontal = imageop.detect_horizontal(single_table_snip, scale=hor_scale)
        ## OLD
        # vertical = imageop.detect_vertical_and_clean_text(single_table_snip, char_dim_cutoff, scale=55)
        ##
        ## NEW
        vertical = imageop.detect_vertical_and_clean_text(single_table_snip, char_dim_cutoff, scale=vert_scale)
        ##
        mask = horizontal + vertical
        joints = cv2.bitwise_and(horizontal, vertical)
        _, thresh = cv2.threshold(mask, 127, 255, 0)
        final_img, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        thresh_cpy = thresh.copy()

        if debug_flag:
            cv2.imwrite(debug_dir + file_name + '_maskHor_' +contour_level+ str(random.randint(1, 101)) + '.png', horizontal)
            cv2.imwrite(debug_dir + file_name + '_maskVert_' +contour_level+ str(random.randint(1, 101)) + '.png', vertical)
            cv2.imwrite(debug_dir + file_name + '_mask_' +contour_level+ str(random.randint(1, 101)) + '.png', mask)
            cv2.imwrite(debug_dir + file_name + '_joints_' +contour_level+ '.png', joints)
            cv2.imwrite(debug_dir + file_name + '_thresh_' +contour_level+ '.png', thresh)
            cv2.imwrite(debug_dir + file_name + '_find_contours_' +contour_level+ '.png', final_img)
            cv2.imwrite(debug_dir + file_name + '_thrsh_cpy_' +contour_level+ '.png', thresh_cpy)

        if debug_flag:
            debug_contour_counter = 0



        for contour in contours:

            all_black = np.zeros((bw.shape[0], bw.shape[1]), dtype=int)
            contours_poly = cv2.approxPolyDP(contour, 3, True)
            bound_rect = cv2.boundingRect(contours_poly)

            x, y, w, h = bound_rect
            roi = joints[y:y + h, x:x + w]



            if debug_flag:
                cv2.rectangle(snip_input_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
                row, col = np.where(joints)
                row, col = list(row), list(col)
                ind = list(zip(row, col))
                for j in ind:
                    cx, cy = j
                    cv2.circle(snip_input_copy, (int(cy), int(cx)), 5, (0, 0, 0), -5)

            if debug_flag:
                cv2.imwrite(debug_dir + file_name + '_roi_' +contour_level+ str(debug_contour_counter) + '.png', roi)
                debug_contour_counter += 1

            '''
            _, thresh1 = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            img_, joints_contours, __ = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if debug_flag:
                cv2.imwrite(debug_dir + file_name + '_find_contours_joints' + str(debug_contour_counter) + '.png', img_)

            if len(joints_contours) <= 0:
                continue
            
            '''

            all_black[y:y + h, x:x + w] = thresh_cpy[y:y + h, x:x + w]

            bbox_out.append((x, y, w, h))
            rois.append(all_black.copy())
            bboxline_coords.append([(x, y, x + w, y),
                                    (x + w, y, x + w, y + h),
                                    (x, y + h, x + w, y + h),
                                    (x, y, x, y + h)])

        # bbox_out = remove_small_bboxs(bbox_out)
        unique_bboxs, flags = eliminate_inner_rect_bbox_thresh(bbox_out)
        unique_bboxline_coords, unique_rois = [], []

        for idx, flag in enumerate(flags):
            if flag != 0:
                unique_bboxline_coords.append(bboxline_coords[idx])
                mask_img = rois[idx]
                [x, y, w, h] = bbox_out[idx]
                all_black = mask_dotted * 0
                pad = 5
                all_black[y - pad:y + h + pad, x - pad:x + w + pad] = mask_dotted[y - pad:y + h + pad,
                                                                                  x - pad:x + w + pad]
                mask_img += all_black
                unique_rois.append(mask_img)
                if debug_flag:
                    cv2.imwrite(debug_dir + file_name + '_thrsh_cpy_' +contour_level+ str(idx) + '.png', rois[idx])

        if debug_flag:
            logger.debug('rois_unique: '+str(unique_rois))
            net = np.zeros(unique_rois[0].shape, dtype=int)
            for p in unique_rois:
                net += p
            cv2.imwrite(debug_dir + file_name + '_thrsh_cpy_net_' +contour_level+ str(random.randint(1, 101)) + '.png', net)
            net_img += net


        tot_unique_bboxs += unique_bboxs
        tot_unique_rois += unique_rois
        tot_unique_bboxline_coords += unique_bboxline_coords
        counter += 1


    if debug_flag:
        cv2.imwrite(debug_dir + file_name + '_thrsh_cpy_net_img_' +contour_level+ str(random.randint(1, 101)) + '.png', net_img)
    return tot_unique_rois, tot_unique_bboxs, tot_unique_bboxline_coords


def eliminate_inner_rect_bbox(tables):

    l = len(tables)
    flag = [1] * l
    epsilon = 5

    for i in range(l):
        if flag[i]:
            for j in range(i + 1, l):
                if flag[j]:
                    x1, y1, w1, h1 = tables[i]
                    x2, y2, w2, h2 = tables[j]
                    con1 = ((x2 >= x1)
                            and (x2 <= x1 + w1)
                            and (x2 + w2 <= x1 + w1 + epsilon)
                            and (y2 >= y1)
                            and (y2 <= y1 + h1)
                            and (y2 + h2 <= y1 + h1 + epsilon))

                    con2 = ((x1 >= x2)
                            and (x1 <= x2 + w2)
                            and (x1 + w1 <= x2 + w2 + epsilon)
                            and (y1 >= y2)
                            and (y1 <= y2 + h2)
                            and (y1 + h1 <= y2 + h2 + epsilon))

                    if con1:
                        flag[j] = 0
                    if con2:
                        flag[i] = 0
                        break
    unique = []
    for k in range(l):
        if flag[k] != 0:
            unique.append(tables[k])
    return unique, flag


def get_overlap_area(bb1, bb2):

    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2

    # determine the coordinates of the intersection rectangle

    x_left = int(max(x1, x2))
    y_top = int(max(y1, y2))
    x_right = int(min(x1 + w1, x2 + w2))
    y_bottom = int(min(y1 + h1, y2 + h2))

    if x_right < x_left or y_bottom < y_top:
        return 0.0, [x_left, y_top, (x_right - x_left), (y_bottom - y_top)]

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both boxes
    bb1_area = w1 * h1
    bb2_area = w2 * h2

    min_bb_area = float(min(bb1_area, bb2_area))

    overlap_area = intersection_area / min_bb_area

    return overlap_area, [x_left, y_top, (x_right - x_left), (y_bottom - y_top)]





def get_overlap_area_tables(tables):

    l = len(tables)
    overlap_areas = np.zeros((l, l), dtype='float')
    overlap_thresh_flags = np.zeros((l, l), dtype='int')
    overlap_bound_rects = np.zeros((l, l, 4), dtype='int')
    rect_areas_img = []

    for table in tables:
        x, y, w, h = table
        table_area = w * h
        rect_areas_img.append(table_area)

    rect_areas_img = np.array(rect_areas_img)

    for i in range(l):
        for j in range(i + 1, l):
            area, rect = get_overlap_area(tables[i], tables[j])
            if area > 0.9:
                flag = 1
            else:
                flag = 0
            overlap_areas[i, j] = area
            overlap_thresh_flags[i, j] = flag
            overlap_bound_rects[i, j] = rect
            overlap_areas[j, i] = area
            overlap_thresh_flags[j, i] = flag
            overlap_bound_rects[j, i] = rect

    return overlap_areas, overlap_thresh_flags, overlap_bound_rects, rect_areas_img

def remove_small_bboxs(bboxs):
    bboxs_big = []
    for bbox in bboxs:
        x,y,w,h = bbox
        if (w>=10) and (h<=10):
            bboxs_big.append(bbox)
    return bboxs_big


def eliminate_inner_rect_bbox_thresh(tables):

    l = len(tables)
    flag = [1] * l

    for i in range(l):
        if flag[i]:
            for j in range(i + 1, l):
                if flag[j]:
                    x1, y1, w1, h1 = tables[i]
                    x2, y2, w2, h2 = tables[j]
                    overlap, _ = get_overlap_area(tables[i],tables[j])
                    if overlap > 0.9:
                        if((w1*h1) >= (w2*h2)):
                            flag[j] = 0
                        else:
                            flag[i] = 0
                            break

    unique = []
    for k in range(l):
        if flag[k] != 0:
            unique.append(tables[k])
    return unique, flag