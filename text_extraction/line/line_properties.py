from apporchid.common.logger import logger
import numpy as np

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def slope(line):
    x1, y1, x2, y2 = 0, 1, 2, 3
    try:
        if line[x2] != line[x1]:
            s = abs(line[y2] - line[y1]) / abs(line[x2] - line[x1])
        else:
            s = float('inf')
    except Exception as e:
        logger.debug(e)
        s = float('inf')
    
    return s
        
        
def intersection(s1, s2):
    segment_endpoints = []
    left = max(min(s1[0], s1[2]), min(s2[0], s2[2]))
    right = min(max(s1[0], s1[2]), max(s2[0], s2[2]))
    top = max(min(s1[1], s1[3]), min(s2[1], s2[3]))
    bottom = min(max(s1[1], s1[3]), max(s2[1], s2[3]))

    if top > bottom or left > right:
        segment_endpoints = []
        return ([False, 0])

    elif top == bottom and left == right:
        segment_endpoints.append(left)
        segment_endpoints.append(top)
        return ([True, segment_endpoints, "point"])

    else:
        segment_endpoints.append(left)
        segment_endpoints.append(bottom)
        segment_endpoints.append(right)
        segment_endpoints.append(top)
        return ([True, segment_endpoints, "segment"])
    
    
def line_length(line):
    x1, y1, x2, y2 = 0, 1, 2, 3
    return np.sqrt(np.power(int(line[y2]) - int(line[y1]), 2) + np.power(int(line[x2]) - int(line[x1]), 2))

def intersection_financial_lines(s1, s2):
   segment_endpoints = []
   # left = max(min(s1[0], s1[2]), min(s2[0], s2[2]))
   # right = min(max(s1[0], s1[2]), max(s2[0], s2[2]))
   top = s1[3]
   bottom = s2[3]
   if abs(top - bottom) <= 5:
       if (s1[0] <= s2[0] and s2[0] <= s1[2]) or (s1[0] <= s2[2] and s2[2] <= s1[2]):
           return True
       else:
           return False
   else:
       return False