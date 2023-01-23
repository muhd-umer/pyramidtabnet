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
import torch

from mmdet.apis.inference import init_detector, show_result_pyplot

import argparse
from utils import (
    detection_dict,
    get_preds,
)
from termcolor import colored
from contextlib import contextmanager
import sys


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
        "--input",
        help="Path to input image/directory to perform inference on.",
        required=True,
    )
    parser.add_argument(
        "--weights",
        help="Checkpoint file to load weights from.",
        default="weights/table_det.pth",
        required=False,
    )
    parser.add_argument(
        "--output",
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
    parser.add_argument(
        "--save-detections",
        help="Enable/disable saving of visualizations.",
        action=argparse.BooleanOptionalAction,
        required=False,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    assert osp.exists(
        args.input
    ), "Input image/directory does not exist. Recheck passed argument."

    allowed_extensions = ["jpg", "jpeg", "bmp", "png"]
    file_list = []

    if osp.isfile(args.input):
        path, base_name = (
            osp.split(str(args.input))[0],
            osp.split(str(args.input))[1],
        )
        file_list.append(base_name)

    else:
        path = str(args.input)
        file_list = os.listdir(str(args.input))

    input_list = [
        fn for fn in file_list if any(fn.endswith(ext) for ext in allowed_extensions)
    ]

    assert (
        len(input_list) != 0
    ), "Input file(s) must be among the allowed extensions. Allowed Extensions: [jpg, jpeg, bmp, png, webp, tiff]."

    if args.device == "cuda":
        assert torch.cuda.is_available(), f"No CUDA Runtime found."

    checkpoint_file = args.weights
    config_file = args.config_file

    with suppress_stdout():
        table_det = init_detector(
            config_file, checkpoint_file, device=args.device, cfg_options=detection_dict
        )

    print(
        colored(
            "Model loaded successfully.",
            "cyan",
        )
    )

    print(colored(f"Results will be saved to {osp.abspath(args.output)}", "blue"))

    for input in input_list:
        image = cv2.imread(osp.join(path, input), cv2.IMREAD_COLOR)
        image = image[:, :, :3]  # Removing possible alpha channel

        result_tables, tables = get_preds(
            image, table_det, 0.8, axis=0, merge=False, craft=False, device=args.device
        )

        # Exit the inference script if no predictions are made
        if (result_tables, tables) == (0, 0):
            print(
                colored(
                    f"No predictions were made in image: {input}",
                    "red",
                )
            )
            continue

        else:
            save_dir = osp.abspath(args.output)
            base_name = input[:-4]

            if args.save_detections:
                base_name = "bbox_detections"
                save_dir = osp.join(osp.abspath(args.output), input[:-4])

            os.makedirs(save_dir, exist_ok=True)

            # Saving bounding box coordinates in a text file
            file = open(osp.join(save_dir, f"{base_name}.txt"), "w")
            for table in tables:
                file.write(
                    "table "
                    + str(table[0])
                    + " "
                    + str(table[1])
                    + " "
                    + str(table[2])
                    + " "
                    + str(table[3])
                    + "\n"
                )
            file.close()

            print(colored(f"Inference on {input} completed.", "blue"))

            if args.save_detections:  # Saving results
                save_tables = image.copy()

                show_result_pyplot(
                    table_det,
                    image,
                    result_tables,
                    score_thr=0.8,
                    out_file=osp.join(save_dir, "instance_detections.png"),
                )

                for box in tables:
                    save_tables = cv2.rectangle(
                        save_tables,
                        (box[0], box[1]),
                        (box[2], box[3]),
                        (255, 0, 0),
                        3,
                    )

                cv2.imwrite(
                    osp.join(save_dir, "table_detections.png"),
                    save_tables,
                )
