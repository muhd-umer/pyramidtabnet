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

import os
import os.path as osp
import math
import numpy as np
import cv2
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


def get_row_structure(cells, columns):
    """
    Get row structure from detected cells and columns
    Returns: dictionary of format {"cell_bbox": [row_position]
                                    ...                     }
    """
    row_identifiers = []
    columns = sort_by_element(columns, 0)

    for cell in cells:
        if cell_in_column(cell, columns[0]) is not None:
            row_identifiers.append(cell)

    row_identifiers = sort_by_element(row_identifiers, 1)
    row_extension = []

    for cell in row_identifiers:
        row_extension.append(int((cell[1] + cell[3]) / 2))

    cells = sort_by_element(cells, 1)
    row_positions = []
    identified_rows = []
    unidentified_rows = cells.copy()

    for cell in cells:
        for row in row_extension:
            if row_in_box(row, cell) is not None:
                row_positions.append([cell, row_extension.index(row)])
                identified_rows.append(cell)

    unidentified_rows = [x for x in unidentified_rows if x not in identified_rows]
    unidentified_rc = [int((coord[1] + coord[3]) / 2) for coord in unidentified_rows]
    supp_rows = []

    for r in range(len(row_extension)):
        if r == 0:
            for c in range(len(unidentified_rc)):
                if unidentified_rc[c] < row_extension[r]:
                    supp_rows.append(
                        [
                            unidentified_rows[c],
                            r + 0.5,
                        ]
                    )

        if r == len(row_extension) - 1:
            for c in range(len(unidentified_rc)):
                if unidentified_rc[c] > row_extension[r]:
                    supp_rows.append(
                        [
                            unidentified_rows[c],
                            r + 0.5,
                        ]
                    )
        else:
            for c in range(len(unidentified_rc)):
                if (
                    unidentified_rc[c] > row_extension[r]
                    and unidentified_rc[c] < row_extension[r + 1]
                ):
                    supp_rows.append(
                        [
                            unidentified_rows[c],
                            r + 0.5,
                        ]
                    )

    for item in supp_rows:
        row_positions.append(item)

    row_positions = sort_by_element(row_positions, 1)

    for idx in range(len(row_positions)):
        if row_positions[idx][1] % 1 != 0:
            row_positions[idx][1] = math.ceil(row_positions[idx][1])
            indexed_addition(row_positions[idx], idx)

    # Define row structure dictionary
    row_cells = []
    row_index = []

    for item in row_positions:
        row_cells.append(item[0])
        row_index.append(item[1])

    row_structure = {str(item): [] for item in row_cells}

    for item in row_positions:
        row_structure[str(item[0])].append(item[1])

    return row_structure


def get_column_stucture(cells, columns):
    """
    Get column structure from detected cells and columns
    Returns: dictionary of format {"cell_bbox": [col_position]
                                    ...                     }
    """
    cells = sort_by_element(cells, 1)
    col_positions = []
    identified_cols = []
    unidentified_cols = cells.copy()

    for cell in cells:
        for col in columns:
            if cell_in_column(cell, col) is not None:
                col_positions.append([cell, columns.index(col)])
                identified_cols.append(cell)

    unidentified_cols = [x for x in unidentified_cols if x not in identified_cols]
    unidentified_cc = [[int(coord[0]), int(coord[2])] for coord in unidentified_cols]
    supp_cols = []

    for col in range(len(columns)):
        if col == 0:
            for c in range(len(unidentified_cc)):
                if col_in_box(unidentified_cc[c][0], columns[col]) and col_in_box(
                    unidentified_cc[c][1], columns[col]
                ):
                    supp_cols.append(
                        [
                            unidentified_cols[c],
                            col,
                        ]
                    )
                elif col_in_box(unidentified_cc[c][0], columns[col]) or col_in_box(
                    unidentified_cc[c][1], columns[col]
                ):
                    supp_cols.append(
                        [
                            unidentified_cols[c],
                            col,
                        ]
                    )

        elif col == len(columns) - 1:
            for c in range(len(unidentified_cc)):
                if col_in_box(unidentified_cc[c][0], columns[col]) and col_in_box(
                    unidentified_cc[c][1], columns[col]
                ):
                    supp_cols.append(
                        [
                            unidentified_cols[c],
                            col,
                        ]
                    )
                elif col_in_box(unidentified_cc[c][0], columns[col]) or col_in_box(
                    unidentified_cc[c][1], columns[col]
                ):
                    supp_cols.append(
                        [
                            unidentified_cols[c],
                            col,
                        ]
                    )
        else:
            for c in range(len(unidentified_cc)):
                if col_in_box(unidentified_cc[c][0], columns[col]) and col_in_box(
                    unidentified_cc[c][1], columns[col]
                ):
                    supp_cols.append(
                        [
                            unidentified_cols[c],
                            col,
                        ]
                    )
                elif col_in_box(unidentified_cc[c][0], columns[col]) or col_in_box(
                    unidentified_cc[c][1], columns[col]
                ):
                    supp_cols.append(
                        [
                            unidentified_cols[c],
                            col,
                        ]
                    )

    for item in supp_cols:
        col_positions.append(item)

    col_positions = sort_by_element(col_positions, 1)

    # Define column structure dictionary
    col_cells = []
    col_index = []

    for item in col_positions:
        col_cells.append(item[0])
        col_index.append(item[1])

    col_structure = {str(item): [] for item in col_cells}

    for item in col_positions:
        col_structure[str(item[0])].append(item[1])

    return col_structure
