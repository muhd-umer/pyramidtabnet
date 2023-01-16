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

Evaluation script for precision, recall, and F1 score
"""

import warnings

warnings.filterwarnings("ignore")

import os
import os.path as osp
import cv2
import numpy as np
import torch

from mmdet.apis import inference_detector
from mmdet.apis.inference import init_detector

import argparse
from utils import get_area, get_overlap_status, calculate_metrics, pascal_voc_bbox
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


def average(list):
    return sum(list) / len(list)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate your detector.")
    parser.add_argument(
        "--config-file",
        help="Path to detector config file. (In configs/ directories.)",
        required=True,
    )
    parser.add_argument(
        "--input-dir",
        help="Path to test dataset to perform evaluation on.",
        required=True,
    )
    parser.add_argument(
        "--gt-dir",
        help="Path to ground truth files (PASCAL VOC).",
        required=True,
    )
    parser.add_argument(
        "--weights",
        help="Checkpoint file to load weights from.",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        help="Path to output where bounding box predictions are saved.",
        default=None,
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


def get_preds(image, model, thresh):
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
    model = model
    result = inference_detector(model, image)

    det_boxes = []
    conf = []
    remove_index = []

    for r in result[0][0]:
        conf.append(r[4])
        if r[4] > 0.5:
            det_boxes.append(r.astype(int))

    if len(det_boxes) == 0:
        pass
    else:
        det_boxes = np.array(det_boxes)[:, :4]
        det_boxes = det_boxes.tolist()
        pred_boxes = det_boxes.copy()

        for i in range(len(det_boxes)):
            for k in range(len(det_boxes)):
                if (k != i) and (
                    get_overlap_status(det_boxes[i], det_boxes[k]) == True
                ):
                    if (det_boxes[i] in pred_boxes) and (
                        get_area(det_boxes[i]) < get_area(det_boxes[k])
                    ):
                        pred_boxes.remove(det_boxes[i])
                        remove_index.append(i)
                else:
                    pass

    for j in sorted(remove_index, reverse=True):
        del conf[j]

    return result, pred_boxes, conf


if __name__ == "__main__":
    args = parse_args()

    assert osp.exists(
        args.input_dir
    ), "Input directory does not exist. Recheck file directory."

    input_path = osp.abspath(str(args.input_dir))
    gt_path = osp.abspath(str(args.gt_dir))
    file_list = os.listdir(input_path)

    assert len(os.listdir(input_path)) == len(
        os.listdir(gt_path)
    ), "Input directory must contain the same number of files as in the ground truth directory."

    if args.device == "cuda":
        assert torch.cuda.is_available(), f"No CUDA Runtime found."

    # MMDetection Inference Pipeline
    checkpoint_file = args.weights
    config_file = args.config_file

    with suppress_stdout():
        model = init_detector(config_file, checkpoint_file, device=args.device)

    print(
        colored(
            "Model loaded successfully.",
            "cyan",
        )
    )

    precision, recall, f1_score = [], [], []

    for i in range(len(file_list)):
        ori_img = cv2.imread(osp.join(input_path, file_list[i]), cv2.IMREAD_COLOR)
        ori_img = ori_img[:, :, :3]  # Removing possible alpha channel
        result, pred_boxes, conf = get_preds(ori_img, model, 0.8)

        # Exit the inference script if no predictions are made
        if (result, pred_boxes) == (0, 0):
            print(
                colored(
                    f"No predictions were made in image {file_list[i]}",
                    "red",
                )
            )
            pass

        else:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)

                # Saving bounding box coordinates in a text file
                file = open(osp.join(args.output_dir, f"{file_list[i][:-4]}.txt"), "w")
                for k in range(len(pred_boxes)):
                    file.write(
                        "table "
                        + str(conf[k])
                        + " "
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

                print(colored(f"Saved inference on {file_list[i]}.", "blue"))

            for j in range(len(pred_boxes)):
                pred_boxes[j].insert(0, "table")

            gt_boxes = pascal_voc_bbox(osp.join(gt_path, file_list[i][:-4] + ".xml"))
            p_item, r_item, f1_item = calculate_metrics(pred_boxes, gt_boxes)

            precision.append(p_item)
            recall.append(r_item)
            f1_score.append(f1_item)

    # Print metrics to console
    print(
        colored(
            f"Precision: {average(precision)}",
            "blue",
        )
    )
    print(
        colored(
            f"Recall: {average(recall)}",
            "blue",
        )
    )
    print(
        colored(
            f"F1-Score: {average(f1_score)}",
            "blue",
        )
    )
