"""
MIT License:
Copyright (c) 2022 Muhammad Umer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

Utility functions to manipulate table structure
"""

import math
import xml.etree.ElementTree as ET


def point_in_box(point, box):
    """
    Check if a point lies within a bounding box
    """
    if (
        box[0] <= point[0]
        and point[0] <= box[2]
        and box[1] <= point[1]
        and point[1] <= box[3]
    ):
        return True


def cell_in_column(cell_box, column_box):
    """
    Check if a cell lies within a column
    """
    cell_point = (int(cell_box[0]), int(int((cell_box[1] + cell_box[3]) / 2)))
    return point_in_box(cell_point, column_box)


def sort_by_element(list, element):
    return sorted(list, key=lambda x: x[element])


def row_in_box(y, box):
    """
    Check if a y point lies within a bounding box
    """
    if box[1] <= y and y <= box[3]:
        return True


def col_in_box(x, box):
    """
    Check if a x point lies within a bounding box
    """
    if box[0] <= x and x <= box[2]:
        return True


def indexed_addition(position_list, index):
    """
    Add a value to all elements after an index
    """
    for idx in range(len(position_list)):
        if idx > index:
            position_list[idx][1] += 1

    return position_list


def columns_to_lines(column_boxes):
    """
    Converts bounding boxes of [xmin, ymin, xmax, ymax] format
    to [xmin, xmax]
    """
    return [[column[0], column[2]] for column in column_boxes]


def column_mapping(columns, cells):
    """
    Get column structure from detected cells and columns
    Returns: dictionary of format {"cell_bbox": [col_position]
                                    ...                     }
    """
    column_mapping = {}

    for cell in cells:
        cell = tuple(cell)
        xmin, _, xmax, _ = cell

        for i in range(len(columns)):
            if i == 0:
                if xmin <= columns[i][1]:
                    column_mapping[cell] = (i, i)
                    break

            elif i == len(columns) - 1:
                if xmax >= columns[i][0]:
                    column_mapping[cell] = (i, i)
                    break

            else:
                if columns[i][0] <= xmin <= columns[i][1]:
                    column_mapping[cell] = (i, i)
                    break
                elif columns[i][1] < xmin < columns[i + 1][0]:
                    column_mapping[cell] = (i, i + 1)
                    break

    return column_mapping


def columns_to_lines(column_boxes):
    """
    Converts [xmin, ymin, xmax, ymax] to [xmin, xmax] lines
    """
    return [[column[0], column[2]] for column in column_boxes]


def count_columns(cells_dict):
    column_count = {}
    column_cells = {}

    for cells, columns in cells_dict.items():
        start_column, end_column = columns

        for column in range(start_column, end_column + 1):
            if column in column_count:
                column_count[column] += 1
                column_cells[column].append(cells[1])
            else:
                column_count[column] = 1
                column_cells[column] = [cells[1]]

    highest_column = max(column_count, key=column_count.get)

    return highest_column, column_cells[highest_column]


def row_mapping(rows, cells):
    """
    Get row structure from detected cells and columns
    Returns: dictionary of format {"cell_bbox": [row_position]
                                    ...                     }
    """
    row_mapping = {}
    sorted_rows = sorted(rows)

    for cell in cells:
        cell = tuple(cell)
        _, ymin, _, ymax = cell
        start_row, end_row = None, None

        for i in range(len(sorted_rows)):
            if start_row is None:
                if ymin <= sorted_rows[i]:
                    start_row = rows.index(sorted_rows[i])
            if ymax > sorted_rows[i]:
                end_row = rows.index(sorted_rows[i])

        if start_row is None:
            start_row = end_row

        if end_row is None:
            end_row = start_row

        row_mapping[cell] = (start_row, end_row)

    return row_mapping
