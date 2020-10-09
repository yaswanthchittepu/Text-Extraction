import pandas as pd
import apporchid.agora.cv.line.line_properties as lineprop

def create_table_json(data):

    data['xdiff'] = abs(data['x1'] - data['x2'])
    data['ydiff'] = abs(data['y1'] - data['y2'])
    #data[(data['x1'] != data['x2']) & (data['y1'] != data['y2'])]

    ###find those lines which are neither parallel to x-axis nor y-axis and fix them
    inclined_lines = data[(data['x1'] != data['x2']) & (data['y1'] != data['y2'])]
    for index, row in inclined_lines.iterrows():
        if row['xdiff'] > 2 and row['ydiff'] > 2:
            continue
        if (row['xdiff'] < row['ydiff']):
            data.iloc[index, data.columns.get_loc('x2')] = row['x1']
        else:
            data.iloc[index, data.columns.get_loc('y2')] = row['y1']

    # split the data into vertical and horizontal lines
    horizontal = data[data['y1'] == data['y2']]
    horizontal = horizontal.drop(['ydiff'], axis=1)
    horizontal = horizontal.sort_values('xdiff')
    horizontal.index = range(len(horizontal))
    vertical = data[data['x1'] == data['x2']]
    vertical = vertical.drop(['xdiff'], axis=1)
    # vertical.columns = ['x1','y2','x2','y1','ydiff']
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
        if (row['lag1'] < 15):
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
            # yvalue = yvalue.iloc(yvalue.index)[0]
            yvalue = yvalue[yvalue.index[0]]
            df = pd.DataFrame(
                [[temp1.x1.min(), yvalue, temp1.x2.max(), yvalue, abs(temp1.x1.min() - temp1.x2.max()), 0, 0]],
                columns=horizontal1.columns, index=[len(horizontal1)])
            horizontal1 = pd.concat([horizontal1, df])
        else:
            for i in temp1.index:
                horizontal1.iloc[i, horizontal1.columns.get_loc('ignore')] = 1
            yvalue = temp1[temp1['xdiff'] == temp1.xdiff.max()].y1
            # yvalue = yvalue.iloc(yvalue.index)[0]
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
        if (row['lag1'] < 15):
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
            # xvalue = xvalue.iloc(xvalue.index)[0]
            xvalue = xvalue[xvalue.index[0]]
            df = pd.DataFrame(
                [[xvalue, temp1.y2.max(), xvalue, temp1.y1.min(), abs(temp1.y1.min() - temp1.y2.max()), 0, 0]],
                columns=vertical1.columns, index=[len(vertical1)])
            vertical1 = pd.concat([vertical1, df])
        else:
            for i in temp1.index:
                vertical1.iloc[i, vertical1.columns.get_loc('ignore')] = 1
            xvalue = temp1[temp1['ydiff'] == temp1.ydiff.max()].x1
            # xvalue = xvalue.iloc(xvalue.index)[0]
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
    ########horizontal2 = horizontal1[horizontal1['ignore']==0]
    # Make y1 < y2 for vertical2 lines
    temp = vertical2[vertical2.y2 < vertical2.y1]
    if len(temp) > 0:
        y1coord = vertical2.columns.get_loc('y1')
        y2coord = vertical2.columns.get_loc('y2')
        for index, row in temp.iterrows():
            vertical2.iloc[index, y1coord] = row['y2']
            vertical2.iloc[index, y2coord] = row['y1']

    ##horizontal2.index = range(len(horizontal2))
    x1coord = horizontal2.columns.get_loc('x1')
    x2coord = horizontal2.columns.get_loc('x2')
    for index, row in horizontal2.iterrows():
        # extend left
        temp = vertical2[(vertical2.y1 - 50 <= row.y1) & (vertical2.y2 + 50 >= row.y1) & (vertical2.x1 < row.x1 + 50)]
        if (len(temp) > 0):
            horizontal2.iloc[index, x1coord] = temp.x1.max()
            # print("#extend left")
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

    # Generate vertices
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
    # vertixdf = vertixdf.sort_values('y1')
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
            topright = (row1.x1, row.y1)  # 653, 635
            ### Ensure topleft and topright are connected horizontally
            ### also include topright to topleft with and below done
            if ((horizontal[((horizontal.y1 >= row.y1 - 10) & (horizontal.y1 <= row.y1 + 10)) & \
                            ((horizontal.x1 <= row.x1 + 10) & (horizontal.x2 > row.x1 + 10))].empty)):
                topright = ""
                break
            ### there exists a linesegment (vertical) downwords rightx.x1, row.y1?
            #             if (vertical[((vertical.x1 >= row1.x1 - 10) & (vertical.x1 <= row1.x1 +10)) & \
            #                          ((vertical.y1 <= row.y1 + 10) & (vertical.y2 > row.y1 + 30))].empty):

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
            #             if (horizontal[((horizontal.y1 >= row2.y1 - 10) & (horizontal.y1 <= row2.y1 + 10)) & \
            #                            ((horizontal.x1 <= row.x1 +50) & (horizontal.x2 > row.x1+0))].empty):
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

    ###########################
    # find tables using boxes
    ###########################
    from operator import itemgetter
    # boxlist = sorted(boxlist, key=itemgetter(0))
    boxlist = sorted(boxlist, key=itemgetter(1, 0))

    ####################YASH ADDITION###############################################
    # boxlist2 = []
    # for i in boxlist:
    #     if abs(i[0][0] - i[1][0]) >= 20 and abs(i[1][1]-i[2][1]) >=20:
    #         boxlist2.append(i)
    # boxlist = boxlist2
    ####################################################################################

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
                if ((box[0] == tbox[3]) | (box[1] == tbox[2])):  # downside box? sharing the line?
                    table.append(box)
                    usedboxes.append(box)
                    break
        if (len(table)):
            tableslist.append(table)
            table = []
    ##############################################################################################################################################################
    # print(tableslist)
    ##############################################################################################################################################################

    tablestring = _as_html(tableslist)
    css_string = _as_styled_html(tableslist)
    #     tablestring = '<data>\n'+tablestring
    #     tablestring = tablestring+'</data>'

    ##########################Yash addition############################################################################
    # print("Box list starts...................................................\n")
    # print(boxlist)
    # boxlist2 = []
    # for i in boxlist:
    #     if abs(i[0][0] - i[1][0]) >= 20 and abs(i[1][1]-i[2][1]) >=20:
    #         boxlist2.append(i)
    # json_tab_string = create_json_object(tableslist, boxlist2)
    #####################################################################################################################

    json_string = as_json(tableslist, boxlist)
    return tablestring, css_string, json_string

def _as_html(tableslist):
    tablestring = ""
    for table in tableslist:
        hline = table[0][0][1]
        # print("<table>")
        tablestring += "<table>"
        tablestring += "\n" + "<tr>"
        # print("<tr>")
        i = 1
        for box in table:
            #    if (box[0][1] != line):
            if (box[0][1] != hline):
                tablestring += "\n" + "</tr>"
                # print("</tr>")
                tablestring += "\n" + "<tr>"
                # print("<tr>")
                hline = box[0][1]
            # print(box[0],box[1],box[2],box[3])
            tablestring += "\n" + "<td>" + "cell" + str(i) + "(" + str(box[0][0]) + "," + str(box[0][1]) + "," + str(
                box[2][0]) + "," + str(box[2][1]) + ")" + "</td>"
            # print("<td>"+"cell"+str(i)+"("+str(box[0][0])+","+str(box[0][1])+","+str(box[2][0])+","+str(box[2][1])+")"+"</td>")
            i += 1
        tablestring += "\n" + "</tr>"
        # print("</tr>")
        tablestring += "\n" + "</table>\n"
        # print("</table>")
    return tablestring

def _as_styled_html(tableslist):
    css_string = "<!DOCTYPE html> \n <html> \n <head> \n <style> \n table, th, td { \n     border: 1px solid black; \n } \n </style> \n </head> \n <body> \n <html>"
    # tablestring = "<table>"
    for table in tableslist:
        hline = table[0][0][1]
        css_string += "\n" + "<table>"
        css_string += "\n" + "<tr>"
        i = 1
        for box in table:
            if (box[0][1] != hline):
                css_string += "\n" + "</tr>"
                css_string += "\n" + "<tr>"
                hline = box[0][1]
            width = abs(box[0][0] - box[2][0]) / 2
            height = abs(box[0][1] - box[2][1]) / 2
            # tablestring += "\n"+ "<td>"+"cell"+str(i)+"("+str(box[0][0])+","+str(box[0][1])+","+str(box[2][0])+","+str(box[2][1])+")"+"</td>"
            # print("<td>"+"cell"+str(i)+"("+str(box[0][0])+","+str(box[0][1])+","+str(box[2][0])+","+str(box[2][1])+")"+"</td>")
            css_string += "\n" + "<td height=\"" + str(height) + "\"; width=\"" + str(width) + "\">" + "cell: " + str(
                i) + "</td>"
            i += 1
        css_string += "\n" + "</tr>"
        css_string += "\n" + "</table>"
        css_string += "\n" + "</br>"
    css_string += "\n" + "</body>\n</html>"
    return css_string



def as_json(tableslist, boxlist):
    boxhor = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])
    boxvert = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])

    for box in boxlist:
        for point_index in range(len(box)):
            x,y = box[point_index]
            box[point_index] = (int(x),int(y))
        df = pd.DataFrame([[box[0][0] + 0, box[0][1] + 0, box[1][0] - 0, box[1][1] + 0]], columns=boxhor.columns,
                          index=[len(boxhor)])
        boxhor = pd.concat([boxhor, df])
        df = pd.DataFrame([[box[1][0] - 0, box[1][1] + 0, box[2][0] - 0, box[2][1] - 0]], columns=boxvert.columns,
                          index=[len(boxvert)])
        boxvert = pd.concat([boxvert, df])
        df = pd.DataFrame([[box[3][0] + 0, box[3][1] - 0, box[2][0] - 0, box[2][1] - 0]], columns=boxhor.columns,
                          index=[len(boxhor)])
        boxhor = pd.concat([boxhor, df])
        df = pd.DataFrame([[box[0][0] + 0, box[0][1] + 0, box[3][0] + 0, box[3][1] - 0]], columns=boxvert.columns,
                          index=[len(boxvert)])
        boxvert = pd.concat([boxvert, df])

    # Create JSON with span information #
    #####################################
    tableslistd = []  # list of table dicts
    tabledict = {}  # id, tr
    rowdict = {}  # id, td
    coldict = {}  # id, bbox, ocr
    rowslist = []  # list of rows
    collist = []  # list of boxes on the same hline
    boxvert1 = boxvert.drop_duplicates(subset=None, keep='first', inplace=False).copy()
    boxhor1 = boxhor.drop_duplicates(subset=None, keep='first', inplace=False).copy()

    # Make x1 < x2 for horizontal lines
    temp = boxhor[boxhor.x2 < boxhor.x1]
    if len(temp) > 0:
        x1coord = boxhor.columns.get_loc('x1')
        x2coord = boxhor.columns.get_loc('x2')
        for index, row in temp.iterrows():
            boxhor.iloc[index, x1coord] = row['x2']
            boxhor.iloc[index, x2coord] = row['x1']
    # Make y1 < y2 for vertical lines
    temp = boxvert1[boxvert1.y2 < boxvert1.y1]
    if len(temp) > 0:
        y1coord = boxvert1.columns.get_loc('y1')
        y2coord = boxvert1.columns.get_loc('y2')
        for index, row in temp.iterrows():
            boxvert1.iloc[index, y1coord] = row['y2']
            boxvert1.iloc[index, y2coord] = row['y1']

    tableno = 0
    rowno = 0
    colno = 0
    newrow = False
    tableslist1 = []
    for table in tableslist:
        # print("##########################")
        # print(table)
        dict2 = {}
        for t in table:
            if dict2.get(t[0][1], None) == None:
                dict2[t[0][1]] = [t]
            else:
                li = dict2[t[0][1]]
                li.append(t)
                dict2[t[0][1]] = li
        # sort keys dict

        table1 = []
        for key in sorted(dict2.keys()):
            dict1 = {}
            li1 = dict2[key]
            for t in li1:
                if dict1.get(t[0][0], None) == None:
                    dict1[t[0][0]] = t
                else:
                    li = dict1[t[0][0]]
                    li.append(t)
                    dict1[t[0][0]] = li
            # print("hello")
            # print(dict1.keys())
            for key1 in sorted(dict1.keys()):
                table1.append(dict1[key1])
        tableslist1.append(table1)
    tableslist = tableslist1
    ############################################

    for table in tableslist:
        tableno += 1
        rowno += 1
        colno += 1
        hline = table[0][0][1]
        tabledict["id"] = "table" + str(tableno)
        rowdict["id"] = "table" + str(tableno) + " row1"
        i = 1
        # sort the tble

        for box in table:
            if (box[0][1] != hline):
                rowdict["td"] = collist
                collist = []
                rowslist.append(rowdict)
                rowdict = {}
                rowno += 1
                colno = 1
                rowdict["id"] = "table" + str(tableno) + " row" + str(rowno)
                hline = box[0][1]
            coldict["id"] = "table" + str(tableno) + " row" + str(rowno) + " col" + str(colno)
            coldict["table"] = tableno
            coldict["row"] = rowno
            coldict["col"] = colno			
            # box[0],box[1] #h top line
            # box[1],box[2] #v right line
            # box[2],box[3] #h bottom line
            # box[0],box[3] #v left line
            # logic for colspan and row span
            # colspan; no. of vertical lines intersecting -1 with bottom line
            colspan = -1
            temp = boxvert1[(boxvert1.y1 == box[2][1]) & (boxvert1.y2 > box[2][1])]
            for vindex, vrow in temp.drop_duplicates(["x1", "y1"]).iterrows():

                result = lineprop.intersection([box[2][0], box[2][1], box[3][0], box[3][1]],
                                      [vrow.x1, vrow.y1, vrow.x2, vrow.y2])
                # print(box)
                # print([box[2][0],box[2][1],box[3][0],box[3][1]], [vrow.x1,vrow.y1,vrow.x2,vrow.y2])
                if result[0]:
                    colspan += 1
            # rowspan; no. of horizontal lines intersecting -1 with right line
            rowspan = -1
            temp = boxhor1[(boxhor1.x1 == box[1][0]) & (boxhor1.x2 > box[1][0])]
            for hindex, hrow in temp.drop_duplicates(["x1", "y1"]).iterrows():
                result = lineprop.intersection([box[1][0], box[1][1], box[2][0], box[2][1]],
                                      [hrow.x1, hrow.y1, hrow.x2, hrow.y2])
                if result[0]:
                    rowspan += 1

            coldict["bbox"] = str(box[0][0]) + "," + str(box[0][1]) + "," + str(box[2][0]) + "," + str(box[2][1])
            coldict["ocr"] = ""
            coldict["colspan"] = colspan if (colspan != -1 and colspan != 0) else 1
            coldict["rowspan"] = rowspan if (rowspan != -1 and rowspan != 0) else 1
            collist.append(coldict)
            coldict = {}
            colno += 1
            i += 1
        rowdict["td"] = collist
        collist = []
        rowslist.append(rowdict)
        rowdict = {}
        tabledict["tr"] = rowslist
        rowslist = []
        tableslistd.append(tabledict)
        tabledict = {}
        rowno = 0
        colno = 0

    return tableslistd


