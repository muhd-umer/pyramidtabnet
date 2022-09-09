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

Utility functions to manipulate and work with bounding boxes of format:
                      [xmin, ymin, xmax, ymax]
"""

import os
import os.path as osp
import shutil
from xml.dom.minidom import Document
import numpy as np
import cv2
from evaluation import calc_table_score


def generate_xml(output_dir, filenames, outputs_dict, threshold):
    """
    Creates {file_name}-results.xml' for all test images with
    Coords in the format of cTDaR data structure.
    """
    output_dir = osp.join(output_dir, "xml_results")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for file_name in filenames:
        boxes = []
        for r in outputs_dict[file_name][0]:
            if r[4] > threshold:
                boxes.append(r.astype(int))

        boxes = np.array(boxes)[:, :4]
        boxes = boxes.tolist()

        if len(boxes) == 0:
            continue

        # To enable sorting, has no effect on the results
        # boxes.sort(key=lambda x: x[1])

        doc = Document()
        root = doc.createElement("document")
        root.setAttribute("filename", file_name)
        doc.appendChild(root)

        for table_id, bbox in enumerate(boxes, start=1):
            nodeManager = doc.createElement("table")
            nodeManager.setAttribute("id", str(table_id))

            bbox_str = "{},{} {},{} {},{} {},{}".format(
                bbox[0],
                bbox[1],
                bbox[0],
                bbox[3],
                bbox[2],
                bbox[3],
                bbox[2],
                bbox[1],
            )
            nodeCoords = doc.createElement("Coords")
            nodeCoords.setAttribute("points", bbox_str)
            nodeManager.appendChild(nodeCoords)
            root.appendChild(nodeManager)

        xml_filename = f"{file_name[:-4]}-result.xml"
        fp = open(os.path.join(output_dir, xml_filename), "w")
        doc.writexml(fp, indent="", addindent="\t", newl="\n", encoding="utf-8")
        fp.flush()
        fp.close()

    return output_dir


def evaluate_table(xml_dir):
    """
    Returns weighted F1 score
    """
    results = calc_table_score(xml_dir)

    return results["wF1"]


def get_overlap_status(R1, R2):
    """
    Returns true if there is an ovevrlap between
    the two argument bounding boxes
    """
    if (R1[0] >= R2[2]) or (R1[2] <= R2[0]) or (R1[3] <= R2[1]) or (R1[1] >= R2[3]):
        return False
    else:
        return True


def get_height(box):
    """
    Returns height of bbox
    """
    return box[3] - box[1]


def get_width(box):
    """
    Returns width of bbox
    """
    return box[2] - box[0]


def get_area(box):
    """
    Returns area of bbox in float
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def get_nearness(a, b, factor):
    """
    Returns true if values are near within
    a specified value (factor)
    """
    difference = abs(a - b)

    if difference == 0 or difference < factor:
        return True

    else:
        return False

def get_bbox(table_mask):
    """
    Get bounding box coordinates from a 
        mask, we filter out contours
        with area less than 50 so
        noise isnt marked as a mask
    """
    table_mask = table_mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    table_contours = []
    boxes = []

    for c in contours:
        if cv2.contourArea(c) > 50:
            table_contours.append(c)

    if len(table_contours) == 0:
        return None

    table_boundRect = [None] * len(table_contours)

    for i, c in enumerate(table_contours):
        polygon = cv2.approxPolyDP(c, 1, True)
        table_boundRect[i] = cv2.boundingRect(polygon)

    table_boundRect.sort()

    for x, y, w, h in table_boundRect:
        boxes.append([x, y, x + w, y + h])

    return boxes


def get_center(bbox):
    """
    Returns the center of bbox
    """
    xCenter = round((bbox[0] + bbox[2]) / 2)
    yCenter = round((bbox[1] + bbox[3]) / 2)

    return xCenter, yCenter


def expand_mask(mask, bbox_list):
    """
    Expands masl width to remove extra 
        unnecessary text
    Helps model filtering out text
    """
    for i in range(len(bbox_list)):
        xmin, ymin, xmax, ymax = bbox_list[i]
        xmin, ymin, xmax, ymax = xmin, ymin + 3, xmax, ymax - 3
        h, w = mask.shape[:2]
        mask[ymin:ymax, 0 : xmin + 30] = 255
        mask[ymin:ymax, xmax - 30 : w] = 255

        return cv2.bitwise_not(mask)


def get_paste_location(mask):
    """
    Get the paste location of the external
        table
    Final step in generating semi-new images
    """
    loc_bbox = get_bbox(mask)
    area_bbox = list()
    for i in loc_bbox:
        area_bbox.append(get_area(i))

    area_bbox = list(dict.fromkeys(area_bbox))
    paste_location = loc_bbox[area_bbox.index(max(area_bbox))]

    return paste_location