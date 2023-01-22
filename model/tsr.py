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

Inference Script for Table Structure Recognition
"""

import warnings

warnings.filterwarnings("ignore")

import os
import os.path as osp
import cv2
import numpy as np
import torch
import lxml.etree as etree

from mmdet.apis import inference_detector
from mmdet.apis.inference import init_detector

import argparse
from utils import (
    get_area,
    get_overlap_status,
    structure_dict,
    cell_dict,
    get_column_stucture,
    get_row_structure,
)
from sys import exit
from termcolor import colored
from contextlib import contextmanager
import sys, os


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform inference on your table image."
    )
    parser.add_argument(
        "--config-file",
        help="Path to detector config file. (In configs/ directories.)",
        required=True,
    )
    parser.add_argument(
        "--input-img",
        help="Path to input image to perform inference on.",
        required=True,
    )
    parser.add_argument(
        "--weights-dir",
        help="Directory to load weights from.",
        default="weights/",
        required=False,
    )
    parser.add_argument(
        "--output-dir",
        help="Path to output where bounding box predictions are saved.",
        default="output/",
        required=False,
    )
    parser.add_argument(
        "--device",
        help="Choose inference runtime, either CPU or CUDA. (Defaults to CPU)",
        default="cpu",
        required=False,
    )
    args = parser.parse_args()
    return args


def get_preds(image, model, thresh, axis, merge=True):
    """
    Detect boxes in the input image.
    Args:
        model (nn.Module): The loaded detector.
        img (np.ndarray): Loaded image.
        thresh (float): Threshold for the bboxes and masks.
    Returns:
        result (tuple[list] or list): Detection results of
            of the form (bbox, segm)
        pred_boxes (list): Nested list where each element is
            of the form [xmin, ymin, xmax, ymax]
    """
    result = inference_detector(model, image)

    result_boxes = []

    for r in result[0][axis]:
        if r[4] > thresh:
            result_boxes.append(r.astype(int))

    if len(result_boxes) == 0:
        return 0, 0
    else:
        result_boxes = np.array(result_boxes)[:, :4]
        result_boxes = result_boxes.tolist()
        pred_boxes = result_boxes.copy()

        if merge == True:
            for i in range(len(result_boxes)):
                for k in range(len(result_boxes)):
                    if (k != i) and (
                        get_overlap_status(result_boxes[i], result_boxes[k]) == True
                    ):
                        if (result_boxes[i] in pred_boxes) and (
                            get_area(result_boxes[i]) < get_area(result_boxes[k])
                        ):
                            pred_boxes.remove(result_boxes[i])
                    else:
                        pass

    return result, pred_boxes


if __name__ == "__main__":
    args = parse_args()

    assert osp.exists(
        args.input_img
    ), "Input image does not exist. Recheck file directory."

    image = cv2.imread(str(args.input_img), cv2.IMREAD_COLOR)
    path, base_name = (
        osp.split(str(args.input_img))[0],
        osp.split(str(args.input_img))[1],
    )

    if args.device == "cuda":
        assert torch.cuda.is_available(), f"No CUDA Runtime found."

    # MMDetection Inference Pipeline
    ori_img = cv2.imread(str(args.input_img), cv2.IMREAD_COLOR)
    ori_img = ori_img[:, :, :3]  # Removing possible alpha channel
    draw_cells = ori_img.copy()
    draw_cols = ori_img.copy()

    structure_rec = os.path.join(os.path.abspath(args.weights_dir), "structure_rec.pth")
    cell_det = os.path.join(os.path.abspath(args.weights_dir), "cell_det.pth")

    config_file = args.config_file

    with suppress_stdout():
        cell_model = init_detector(
            config_file, cell_det, device=args.device, cfg_options=cell_dict
        )
        structure_model = init_detector(
            config_file, structure_rec, device=args.device, cfg_options=structure_dict
        )

    print(
        colored(
            "Models loaded successfully.",
            "cyan",
        )
    )

    result_cells, cells = get_preds(ori_img, cell_model, 0.5, axis=0)
    result_columns, columns = get_preds(
        ori_img, structure_model, 0.5, axis=0, merge=False
    )
    result_headers, headers = get_preds(
        ori_img, structure_model, 0.5, axis=1, merge=False
    )

    # Exit the inference script if no predictions are made
    if (result_cells, cells) == (0, 0) or (result_columns, columns) == (0, 0):
        print(
            colored(
                f"No predictions were made in image {base_name}",
                "red",
            )
        )
        exit()

    else:
        os.makedirs(osp.join(args.output_dir, base_name[:-4]), exist_ok=True)

        root = etree.Element("document")

        row_structure = get_row_structure(cells, columns)
        col_structure = get_column_stucture(cells, columns)

        for cell in cells:
            cell_write = etree.Element("cell")
            row_info = row_structure[str(cell)]
            col_info = col_structure[str(cell)]
            start_col, start_row, end_col, end_row = (
                min(col_info),
                min(row_info),
                max(col_info),
                max(row_info),
            )

            cell_write.set("start-col", str(start_col))
            cell_write.set("start-row", str(start_row))
            cell_write.set("end-col", str(end_col))
            cell_write.set("end-row", str(end_row))

            c1 = str(cell[0]) + "," + str(cell[1])
            c2 = str(cell[0]) + "," + str(cell[3])
            c3 = str(cell[2]) + "," + str(cell[3])
            c4 = str(cell[2]) + "," + str(cell[1])

            coords = etree.Element("Coords", points=c1 + " " + c2 + " " + c3 + " " + c4)

            cell_write.append(coords)
            root.append(cell_write)

        file = open(osp.join(args.output_dir, base_name[:-4], "structure.xml"), "w")
        file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        file.write(etree.tostring(root, pretty_print=True, encoding="unicode"))
        file.close()

        for cell in cells:
            draw_cells = cv2.rectangle(
                draw_cells,
                (cell[0], cell[1]),
                (cell[2], cell[3]),
                (255, 0, 255),
                2,
            )

        for column in columns:
            draw_cols = cv2.rectangle(
                draw_cols,
                (column[0], column[1]),
                (column[2], column[3]),
                (255, 0, 255),
                2,
            )

        cv2.imwrite(
            osp.join(args.output_dir, base_name[:-4], "cell_detections.png"), draw_cells
        )
        cv2.imwrite(
            osp.join(args.output_dir, base_name[:-4], "column_detections.png"),
            draw_cols,
        )

        print(colored(f"Inference on {base_name} completed.", "blue"))
        print(colored(f"Results saved at {osp.abspath(args.output_dir)}", "blue"))
