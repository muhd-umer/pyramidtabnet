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

Inference Script to Detect Tables - Refactored with black
Outputs original image with bounding boxes as well as a text file containing
bounding box coordinates. Cropped table images are also saved.
"""

import warnings

warnings.filterwarnings("ignore")

import os
import os.path as osp
import cv2
import numpy as np
import torch

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.apis.inference import init_detector

import argparse
from utils import get_area, get_overlap_status
from sys import exit
from termcolor import colored


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform inference on your table detector."
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
        "--det-weights",
        help="Checkpoint file to load weights from.",
        required=True,
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


def detect_tables(image, model, thresh):
    """
    Detect tables in the input image.
    Args:
        model (nn.Module): The loaded detector.
        img (np.ndarray): Loaded image.
        thresh (float): Threshold for the bboxes and masks.
    Returns:
        result (tuple[list] or list): Detection results of
            of the form (bbox, segm)
        table_boxes (list): Nested list where each element is
            of the form [xmin, ymin, xmax, ymax]
    """
    model = model
    result = inference_detector(model, image)

    det_boxes = []

    for r in result[0][0]:
        if r[4] > thresh:
            det_boxes.append(r.astype(int))

    if len(det_boxes) == 0:
        return 0, 0
    else:
        det_boxes = np.array(det_boxes)[:, :4]
        det_boxes = det_boxes.tolist()
        table_boxes = det_boxes.copy()

        for i in range(len(det_boxes)):
            for k in range(len(det_boxes)):
                if (k != i) and (
                    get_overlap_status(det_boxes[i], det_boxes[k]) == True
                ):
                    if (det_boxes[i] in table_boxes) and (
                        get_area(det_boxes[i]) < get_area(det_boxes[k])
                    ):
                        table_boxes.remove(det_boxes[i])
                else:
                    pass

    return result, table_boxes


if __name__ == "__main__":
    args = parse_args()

    assert osp.exists(
        args.input_img
    ), "Input image does not exist. Recheck file directory."

    image = cv2.imread(str(args.input_img), cv2.IMREAD_COLOR)
    path, base_name = (
        os.path.split(str(args.input_img))[0],
        os.path.split(str(args.input_img))[1],
    )

    if args.device == "cuda":
        assert torch.cuda.is_available(), f"No CUDA Runtime found."

    # MMDetection Inference Pipeline
    ori_img = cv2.imread(str(args.input_img), cv2.IMREAD_COLOR)
    ori_img = ori_img[:, :, :3]  # Removing possible alpha channel

    checkpoint_file = args.det_weights
    config_file = args.config_file

    model = init_detector(config_file, checkpoint_file, device=args.device)
    result, table_boxes = detect_tables(ori_img, model, 0.8)

    # Exit the inference script if no tables are detected.
    if (result, table_boxes) == (0, 0):
        print(
            colored(
                f"No tables were detected in image {base_name}",
                "red",
            )
        )
        exit()

    else:
        os.makedirs(osp.join(args.output_dir, base_name[:-4]), exist_ok=True)

        # Saving bounding box coordinates in a text file
        file = open(osp.join(args.output_dir, base_name[:-4], "bbox_coords.txt"), "w")
        for k in range(len(table_boxes)):
            file.write(
                "table "
                + str(table_boxes[k][0])
                + " "
                + str(table_boxes[k][1])
                + " "
                + str(table_boxes[k][2])
                + " "
                + str(table_boxes[k][3])
                + "\n"
            )
        file.close()

        print(colored(f"Inference on {base_name} completed.", "blue"))
        print(colored(f"Results saved at {osp.abspath(args.output_dir)}", "blue"))

        # Saving result images
        show_result_pyplot(
            model,
            ori_img,
            result,
            score_thr=0.8,
            out_file=osp.join(
                args.output_dir, base_name[:-4], "instance_detections.png"
            ),
        )

        for i in range(len(table_boxes)):
            ori_img = cv2.rectangle(
                ori_img,
                (table_boxes[i][0], table_boxes[i][1]),
                (table_boxes[i][2], table_boxes[i][3]),
                (255, 0, 255),
                2,
            )

        cv2.imwrite(
            osp.join(args.output_dir, base_name[:-4], "bbox_detections.png"), ori_img
        )
