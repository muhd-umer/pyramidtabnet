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

End-to-end script for table recognition
"""

import warnings

warnings.filterwarnings("ignore")

import os
import os.path as osp
import cv2
import torch

from mmdet.apis import inference_detector
from mmdet.apis.inference import init_detector
import lxml.etree as etree

import argparse
from sys import exit
from termcolor import colored
from contextlib import contextmanager
import sys, os

from model import (
    detection_dict,
    structure_dict,
    cell_dict,
    get_column_stucture,
    get_row_structure,
    get_preds,
)


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

    table_det = os.path.join(os.path.abspath(args.weights_dir), "ptn_detection.pth")
    structure_rec = os.path.join(
        os.path.abspath(args.weights_dir), "ptn_recognition.pth"
    )
    cell_det = os.path.join(os.path.abspath(args.weights_dir), "ptn_cells.pth")

    config_file = args.config_file

    with suppress_stdout():
        det_model = init_detector(
            config_file, table_det, device=args.device, cfg_options=detection_dict
        )
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
        save_tables = image.copy()

        result_tables, tables, conf = get_preds(
            image, det_model, 0.8, axis=0, confidence=True
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
            save_dir = osp.join(osp.abspath(args.output), input[:-4])
            os.makedirs(save_dir, exist_ok=True)

            # Saving bounding box coordinates in a text file
            file = open(osp.join(save_dir, "table_coords.txt"), "w")
            for outer_idx in range(len(tables)):
                file.write(
                    "table "
                    + str(conf[outer_idx])
                    + " "
                    + str(tables[outer_idx][0])
                    + " "
                    + str(tables[outer_idx][1])
                    + " "
                    + str(tables[outer_idx][2])
                    + " "
                    + str(tables[outer_idx][3])
                    + "\n"
                )

                save_tables = cv2.rectangle(
                    save_tables,
                    (tables[outer_idx][0], tables[outer_idx][1]),
                    (tables[outer_idx][2], tables[outer_idx][3]),
                    (255, 0, 0),
                    2,
                )

                cropped_table = image[
                    tables[outer_idx][1] : tables[outer_idx][3],
                    tables[outer_idx][0] : tables[outer_idx][2],
                ]
                cv2.imwrite(
                    osp.join(save_dir, f"table_{outer_idx}.png"),
                    cropped_table,
                )

            file.close()
            cv2.imwrite(
                osp.join(save_dir, "table_detections.png"),
                save_tables,
            )

        root = etree.Element("document")

        for inner_idx in range(len(tables)):
            table_image = cv2.imread(osp.join(save_dir, f"table_{inner_idx}.png"))
            table_image = table_image[:, :, :3]  # Removing possible alpha channel
            save_cells = table_image.copy()
            save_columns = table_image.copy()

            result_cells, cells = get_preds(
                table_image, cell_model, 0.3, axis=0, craft=True, device=args.device
            )
            result_columns, columns = get_preds(
                table_image,
                structure_model,
                0.3,
                axis=0,
                merge=False,
                device=args.device,
            )

            tableXML = etree.Element("table")
            tabelCoords = etree.Element(
                "Coords",
                points=str(tables[inner_idx][0])
                + ","
                + str(tables[inner_idx][1])
                + " "
                + str(tables[inner_idx][2])
                + ","
                + str(tables[inner_idx][3])
                + " "
                + str(tables[inner_idx][2])
                + ","
                + str(tables[inner_idx][3])
                + " "
                + str(tables[inner_idx][2])
                + ","
                + str(tables[inner_idx][1]),
            )
            tableXML.append(tabelCoords)

            # Exit the inference script if no cells are detected.
            if (result_cells, cells) == (0, 0) or (result_columns, columns) == (0, 0):
                print(
                    colored(
                        f"No cells were detected in image: table_{outer_idx}.png",
                        "red",
                    )
                )
                exit()

            else:
                row_structure = get_row_structure(cells, columns)
                col_structure = get_column_stucture(cells, columns)

                if row_structure == {} or col_structure == {}:
                    print("Failed to fetch table structure.")

                for cell in cells:
                    cellXML = etree.Element("cell")

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

                    cellXML.set("start-col", str(start_col))
                    cellXML.set("start-row", str(start_row))
                    cellXML.set("end-col", str(end_col))
                    cellXML.set("end-row", str(end_row))

                    c1 = (
                        str(cell[0] + tables[inner_idx][0])
                        + ","
                        + str(cell[1] + tables[inner_idx][1])
                    )
                    c2 = (
                        str(cell[0] + tables[inner_idx][0])
                        + ","
                        + str(cell[3] + tables[inner_idx][1])
                    )
                    c3 = (
                        str(cell[2] + tables[inner_idx][0])
                        + ","
                        + str(cell[3] + tables[inner_idx][1])
                    )
                    c4 = (
                        str(cell[2] + tables[inner_idx][0])
                        + ","
                        + str(cell[1] + tables[inner_idx][1])
                    )

                    coords = etree.Element(
                        "Coords", points=c1 + " " + c2 + " " + c3 + " " + c4
                    )

                    cellXML.append(coords)
                    tableXML.append(cellXML)

                root.append(tableXML)

                for cell in cells:
                    save_cells = cv2.rectangle(
                        save_cells,
                        (cell[0], cell[1]),
                        (cell[2], cell[3]),
                        (255, 0, 0),
                        2,
                    )

                for column in columns:
                    save_columns = cv2.rectangle(
                        save_columns,
                        (column[0], column[1]),
                        (column[2], column[3]),
                        (255, 0, 0),
                        2,
                    )

                cv2.imwrite(
                    osp.join(save_dir, f"table_{inner_idx}_cells.png"),
                    save_cells,
                )
                cv2.imwrite(
                    osp.join(save_dir, f"table_{inner_idx}_columns.png"),
                    save_columns,
                )

        file = open(osp.join(save_dir, "structure.xml"), "w")
        file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        file.write(etree.tostring(root, pretty_print=True, encoding="unicode"))
        file.close()

        print(colored(f"Inference on {base_name} completed.", "blue"))
