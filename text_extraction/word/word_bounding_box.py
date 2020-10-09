import cv2

def word_box_contours(img_dilation, image):

    im2, ctrs, hier = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ctrs = [i for i in ctrs if len(i) < 600]

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    rectangle_coordinates = []
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = image[y:y + h, x:x + w]

        # show ROI
        # cv2.imshow('segment no:'+str(i),roi)
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 3)
        rectangle_coordinates.append((x, y, x + w, y + h))

    return rectangle_coordinates