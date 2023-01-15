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

End-to-end script for table analysis
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
        description="Perform inference on your table detector."
    )
    parser.add_argument(
        "--config-file",
        help="Path to detector config file. (In configs/ directories.)",
        required=True,
    )
    parser.add_argument(
        "--weights-dir",
        help="Directory to load weights from.",
        required=True,
    )
    parser.add_argument(
        "--input-img",
        help="Path to input image to perform inference on.",
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


def get_overlap_status(R1, R2):
    """
    Returns true if there is an ovevrlap between
    the two argument bounding boxes
    """
    if (R1[0] >= R2[2]) or (R1[2] <= R2[0]) or (R1[3] <= R2[1]) or (R1[1] >= R2[3]):
        return False
    else:
        return True


def get_area(box):
    """
    Returns area of bbox in float
    """
    return (box[2] - box[0]) * (box[3] - box[1])


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
    result = inference_detector(model, image)

    result_boxes = []

    for r in result[0][0]:
        if r[4] > thresh:
            result_boxes.append(r.astype(int))

    if len(result_boxes) == 0:
        return 0, 0
    else:
        result_boxes = np.array(result_boxes)[:, :4]
        result_boxes = result_boxes.tolist()
        pred_boxes = result_boxes.copy()

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
        os.path.split(str(args.input_img))[0],
        os.path.split(str(args.input_img))[1],
    )

    if args.device == "cuda":
        assert torch.cuda.is_available(), f"No CUDA Runtime found."

    # MMDetection Inference Pipeline
    ori_img = cv2.imread(str(args.input_img), cv2.IMREAD_COLOR)
    ori_img = ori_img[:, :, :3]  # Removing possible alpha channel
    draw_img = ori_img.copy()

    weights_index = os.listdir(str(args.weights_dir))
    det_weights, struct_weights = weights_index[0], weights_index[1]
    path_to_det = os.path.join(os.path.abspath(args.weights_dir), det_weights)
    path_to_struct = os.path.join(os.path.abspath(args.weights_dir), struct_weights)

    config_file = args.config_file

    with suppress_stdout():
        det_model = init_detector(config_file, path_to_det, device=args.device)
        struct_model = init_detector(config_file, path_to_struct, device=args.device)

    print(
        colored(
            "Models loaded successfully.",
            "cyan",
        )
    )
    table_result, table_boxes = get_preds(ori_img, det_model, 0.5)

    # Exit the inference script if no tables are detected.
    if (table_result, table_boxes) == (0, 0):
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
        file = open(osp.join(args.output_dir, base_name[:-4], "table_coords.txt"), "w")
        for i in range(len(table_boxes)):
            file.write(
                "table "
                + str(table_boxes[i][0])
                + " "
                + str(table_boxes[i][1])
                + " "
                + str(table_boxes[i][2])
                + " "
                + str(table_boxes[i][3])
                + "\n"
            )

            draw_img = cv2.rectangle(
                draw_img,
                (table_boxes[i][0], table_boxes[i][1]),
                (table_boxes[i][2], table_boxes[i][3]),
                (255, 0, 255),
                2,
            )

            cropped_table = ori_img[
                table_boxes[i][1] : table_boxes[i][3],
                table_boxes[i][0] : table_boxes[i][2],
            ]
            cv2.imwrite(
                osp.join(args.output_dir, base_name[:-4], f"table_{i}.png"),
                cropped_table,
            )

        file.close()
        cv2.imwrite(
            osp.join(args.output_dir, base_name[:-4], "table_detections.png"), draw_img
        )

    for k in range(len(table_boxes)):
        table_img = cv2.imread(
            osp.join(args.output_dir, base_name[:-4], f"table_{k}.png"),
            cv2.IMREAD_COLOR,
        )
        table_img = table_img[:, :, :3]  # Removing possible alpha channel
        draw_cells = table_img.copy()

        struct_result, struct_boxes = get_preds(table_img, struct_model, 0.3)

        # Exit the inference script if no cells are detected.
        if (struct_result, struct_boxes) == (0, 0):
            print(
                colored(
                    f"No cells were detected in image table_{i}.png",
                    "red",
                )
            )
            exit()

        else:
            # Saving cell coordinates in a text file
            file = open(
                osp.join(args.output_dir, base_name[:-4], f"cell_coords_{k}.txt"), "w"
            )
            for j in range(len(struct_boxes)):
                file.write(
                    "cell "
                    + str(struct_boxes[j][0])
                    + " "
                    + str(struct_boxes[j][1])
                    + " "
                    + str(struct_boxes[j][2])
                    + " "
                    + str(struct_boxes[j][3])
                    + "\n"
                )

                draw_cells = cv2.rectangle(
                    draw_cells,
                    (struct_boxes[j][0], struct_boxes[j][1]),
                    (struct_boxes[j][2], struct_boxes[j][3]),
                    (255, 0, 255),
                    2,
                )

            file.close()
            cv2.imwrite(
                osp.join(args.output_dir, base_name[:-4], f"table_{k}_structure.png"),
                draw_cells,
            )

print(colored(f"Inference on {base_name} completed.", "blue"))
print(colored(f"Results saved at {osp.abspath(args.output_dir)}", "blue"))