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
import torch
import lxml.etree as etree

from mmdet.apis.inference import init_detector

import argparse
from utils import (
    structure_dict,
    cell_dict,
    get_column_stucture,
    get_row_structure,
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
        "--structure-weights",
        help="Structure checkpoint file to load weights from.",
        default="weights/ptn_recognition.pth",
        required=False,
    )
    parser.add_argument(
        "--cell-weights",
        help="Cell checkpoint file to load weights from.",
        default="weights/ptn_cells.pth",
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
    parser.add_argument('--save', help="Saves results along with visualization images.", action='store_true')
    parser.add_argument('--quiet', help="Perform inference with minimal console output.", action='store_true')
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

    cell_det = args.cell_weights
    structure_rec = args.structure_weights

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

    print(colored(f"Results will be saved to {osp.abspath(args.output)}", "blue"))

    for input in input_list:
        image = cv2.imread(osp.join(path, input), cv2.IMREAD_COLOR)
        image = image[:, :, :3]  # Removing possible alpha channel
        
        result_cells, cells = get_preds(
            image, cell_model, 0.5, axis=0, craft=True, device=args.device
        )
        result_columns, columns = get_preds(
            image, structure_model, 0.5, axis=0, merge=False, device=args.device
        )

        # Exit the inference script if no predictions are made
        if (result_cells, cells) == (0, 0) or (result_columns, columns) == (0, 0):
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

            if args.save:
                base_name = "table_structure"
                save_dir = osp.join(osp.abspath(args.output), input[:-4])

            os.makedirs(save_dir, exist_ok=True)

            root = etree.Element("document")

            row_structure = get_row_structure(cells, columns)
            col_structure = get_column_stucture(cells, columns)

            if row_structure == {} or col_structure == {}:
                print("Failed to fetch table structure.")

            for cell in cells:
                cell_write = etree.Element("cell")

                try:
                    row_info = row_structure[str(cell)]
                    col_info = col_structure[str(cell)]

                except KeyError:
                    continue

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

                coords = etree.Element(
                    "Coords", points=c1 + " " + c2 + " " + c3 + " " + c4
                )

                cell_write.append(coords)
                root.append(cell_write)

            file = open(osp.join(save_dir, f"{base_name}.xml"), "w")
            file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            file.write(etree.tostring(root, pretty_print=True, encoding="unicode"))
            file.close()

            if not args.quiet:
                print(colored(f"Inference on {input} completed.", "blue"))

            if args.save:  # Saving results
                save_cells = image.copy()
                save_columns = image.copy()

                for cell in cells:
                    save_cells = cv2.rectangle(
                        save_cells,
                        (cell[0], cell[1]),
                        (cell[2], cell[3]),
                        (255, 0, 255),
                        2,
                    )

                for column in columns:
                    save_columns = cv2.rectangle(
                        save_columns,
                        (column[0], column[1]),
                        (column[2], column[3]),
                        (255, 0, 255),
                        2,
                    )

                cv2.imwrite(osp.join(save_dir, "cell_detections.png"), save_cells)
                cv2.imwrite(
                    osp.join(save_dir, "column_detections.png"),
                    save_columns,
                )
