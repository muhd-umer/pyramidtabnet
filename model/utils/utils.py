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
import xml.etree.ElementTree as ET


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

    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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


def calculate_iou(box1, box2):
    """
    Computes the IoU of two boxes
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area


def calculate_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Computes precision, recall, and f1 score
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred_box in pred_boxes:
        pred_class, xmin, ymin, xmax, ymax = pred_box
        best_iou = iou_threshold
        best_gt = None
        for gt_box in gt_boxes:
            gt_class, xmin_gt, ymin_gt, xmax_gt, ymax_gt = gt_box
            if pred_class != gt_class:
                continue
            iou = calculate_iou(
                (xmin, ymin, xmax, ymax), (xmin_gt, ymin_gt, xmax_gt, ymax_gt)
            )
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_box

        if best_gt is None:
            false_positives += 1
        else:
            true_positives += 1
            gt_boxes.remove(best_gt)

    false_negatives = len(gt_boxes)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def pascal_voc_bbox(path):
    """
    Extract all table bounding boxes from GT
    """
    tree = ET.parse(path)
    root = tree.getroot()

    bbox_coordinates = []
    for member in root.findall("object"):
        class_name = member[0].text  # class name
        xmin = int(member[4][0].text)
        ymin = int(member[4][1].text)
        xmax = int(member[4][2].text)
        ymax = int(member[4][3].text)

        bbox_coordinates.append([class_name, xmin, ymin, xmax, ymax])

        return bbox_coordinates


detection_dict = {
    "data.train.classes": ("table",),
    "data.val.classes": ("table",),
    "data.test.classes": ("table",),
    "model.roi_head.bbox_head.0.num_classes": 1,
    "model.roi_head.bbox_head.1.num_classes": 1,
    "model.roi_head.bbox_head.2.num_classes": 1,
    "model.roi_head.mask_head.num_classes": 1,
}

cell_dict = {
    "data.train.classes": ("cell",),
    "data.val.classes": ("cell",),
    "data.test.classes": ("cell",),
    "data.val.pipeline.1.img_scale": (1333, 800),
    "data.test.pipeline.1.img_scale": (1333, 800),
    "model.roi_head.bbox_head.0.num_classes": 1,
    "model.roi_head.bbox_head.1.num_classes": 1,
    "model.roi_head.bbox_head.2.num_classes": 1,
    "model.roi_head.mask_head.num_classes": 1,
}

structure_dict = {
    "data.train.classes": (
        "table column",
        "table column header",
    ),
    "data.val.classes": (
        "table column",
        "table column header",
    ),
    "data.test.classes": (
        "table column",
        "table column header",
    ),
    "data.train.pipeline.3.policies.0.0.img_scale": [
        (480, 1333),
        (512, 1333),
        (544, 1333),
        (576, 1333),
        (608, 1333),
        (640, 1333),
        (672, 1333),
        (704, 1333),
        (736, 1333),
        (768, 1333),
    ],
    "data.train.pipeline.3.policies.1.0.img_scale": [
        (400, 1333),
        (500, 1333),
        (600, 1333),
    ],
    "data.train.pipeline.3.policies.1.2.img_scale": [
        (480, 1333),
        (512, 1333),
        (544, 1333),
        (576, 1333),
        (608, 1333),
        (640, 1333),
        (672, 1333),
        (704, 1333),
        (736, 1333),
        (768, 1333),
    ],
    "data.val.pipeline.1.img_scale": (768, 600),
    "data.test.pipeline.1.img_scale": (768, 600),
    "model.roi_head.bbox_head.0.num_classes": 2,
    "model.roi_head.bbox_head.1.num_classes": 2,
    "model.roi_head.bbox_head.2.num_classes": 2,
    "model.roi_head.mask_head.num_classes": 2,
}
