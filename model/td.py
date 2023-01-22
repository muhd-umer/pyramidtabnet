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

Inference Script for Table Detection
Outputs original image with bounding boxes as well as a text file containing
bounding box coordinates.
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
from utils import get_area, get_overlap_status, detection_dict
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
        "--weights",
        help="Checkpoint file to load weights from.",
        default="weights/table_det.pth",
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

    checkpoint_file = args.weights
    config_file = args.config_file

    with suppress_stdout():
        model = init_detector(
            config_file, checkpoint_file, device=args.device, cfg_options=detection_dict
        )

    print(
        colored(
            "Model loaded successfully.",
            "cyan",
        )
    )

    result, pred_boxes = get_preds(ori_img, model, 0.8, 0)

    # Exit the inference script if no predictions are made
    if (result, pred_boxes) == (0, 0):
        print(
            colored(
                f"No predictions were made in image {base_name}",
                "red",
            )
        )
        exit()

    else:
        os.makedirs(osp.join(args.output_dir, base_name[:-4]), exist_ok=True)

        # Saving bounding box coordinates in a text file
        file = open(osp.join(args.output_dir, base_name[:-4], "bbox_coords.txt"), "w")
        for k in range(len(pred_boxes)):
            file.write(
                "table "
                + str(pred_boxes[k][0])
                + " "
                + str(pred_boxes[k][1])
                + " "
                + str(pred_boxes[k][2])
                + " "
                + str(pred_boxes[k][3])
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

        for box in pred_boxes:
            ori_img = cv2.rectangle(
                ori_img,
                (box[0], box[1]),
                (box[2], box[3]),
                (255, 0, 255),
                2,
            )

        cv2.imwrite(
            osp.join(args.output_dir, base_name[:-4], "table_detections.png"), ori_img
        )
