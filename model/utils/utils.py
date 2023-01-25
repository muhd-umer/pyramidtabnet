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
from mmdet.apis import inference_detector
from .craft import Craft
import sys, os
from contextlib import contextmanager
import itertools


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


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


def merge_bounding_boxes(boxes, threshold):
    """
    Merge bounding boxes relatively near to each other
    """
    merged_boxes = []
    # Iterate over each bounding box
    for box in boxes:
        # Initialize flag to check if box has been added to a merged box
        added = False
        # Iterate over each merged box
        for i, mbox in enumerate(merged_boxes):
            # Check if box is relatively and vertically near to current merged box
            if (
                abs(box[0] - mbox[2]) <= threshold
                and abs(box[1] - mbox[3]) <= threshold
            ):
                # Update merged box with new coordinates
                merged_boxes[i] = [
                    min(mbox[0], box[0]),
                    min(mbox[1], box[1]),
                    max(mbox[2], box[2]),
                    max(mbox[3], box[3]),
                ]
                added = True
                break
        # If box has not been added to a merged box, add it as a new merged box
        if not added:
            merged_boxes.append(box)

    return merged_boxes


def get_preds(
    image, model, thresh, axis, merge=True, craft=False, device="cuda", confidence=False
):
    """
    Detect boxes in the input image.
    Args:
        model (nn.Module): The loaded detector.
        img (np.ndarray): Loaded image.
        thresh (float): Threshold for the boxes and masks.
    Returns:
        result (tuple[list] or list): Detection results of
            of the form (bbox, segm)
        pred_boxes (list): Nested list where each element is
            of the form [xmin, ymin, xmax, ymax]
    """
    result = inference_detector(model, image)
    cuda_status = True if device == "cuda" else False

    result_boxes = []
    craft_boxes = []
    conf = []
    remove_index = []

    for r in result[0][axis]:
        conf.append(r[4])
        if r[4] > thresh:
            result_boxes.append(r.astype(int))

    if len(result_boxes) == 0:
        return 0, 0
    else:
        result_boxes = np.array(result_boxes)[:, :4]
        result_boxes = result_boxes.tolist()

        if craft == True:
            with suppress_stdout():
                craft = Craft(
                    crop_type="box",
                    text_threshold=0.5,
                    cuda=cuda_status,
                    long_size=max([image.shape[0], image.shape[1]]),
                    weight_path_craft_net="weights/craft_mlt.pth",
                    weight_path_refine_net="weights/craft_refiner.pth",
                )
                craft_result = craft.detect_text(image)

                for box in np.array(craft_result["boxes"]):
                    craft_boxes.append(
                        list(box[0].astype(int)) + list(box[2].astype(int))
                    )

        combined_boxes = result_boxes + craft_boxes
        premerge_boxes = combined_boxes.copy()
        pred_boxes = combined_boxes.copy()

        if merge == True:
            for i in range(len(combined_boxes)):
                for k in range(len(combined_boxes)):
                    if (k != i) and (
                        get_overlap_status(combined_boxes[i], combined_boxes[k]) == True
                    ):
                        if (combined_boxes[i] in premerge_boxes) and (
                            get_area(combined_boxes[i]) < get_area(combined_boxes[k])
                        ):
                            premerge_boxes.remove(combined_boxes[i])
                            remove_index.append(i)
                    else:
                        pass

            merged_boxes = merge_bounding_boxes(
                premerge_boxes, int(image.shape[0] / 50)
            )
            pred_boxes = merged_boxes.copy()

            for i in range(len(merged_boxes)):
                for k in range(len(merged_boxes)):
                    if (k != i) and (
                        get_overlap_status(merged_boxes[i], merged_boxes[k]) == True
                    ):
                        if (merged_boxes[i] in premerge_boxes) and (
                            get_area(merged_boxes[i]) < get_area(merged_boxes[k])
                        ):
                            premerge_boxes.remove(merged_boxes[i])
                            remove_index.append(i)
                    else:
                        pass

        if confidence:
            for j in sorted(remove_index, reverse=True):
                del conf[j]

            return result, pred_boxes, conf

    return result, pred_boxes


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
    "data.val.pipeline.1.img_scale": (1924, 768),
    "data.test.pipeline.1.img_scale": (1024, 768),
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
    "data.val.pipeline.1.img_scale": (1024, 768),
    "data.test.pipeline.1.img_scale": (1024, 768),
    "model.roi_head.bbox_head.0.num_classes": 2,
    "model.roi_head.bbox_head.1.num_classes": 2,
    "model.roi_head.bbox_head.2.num_classes": 2,
    "model.roi_head.mask_head.num_classes": 2,
}
