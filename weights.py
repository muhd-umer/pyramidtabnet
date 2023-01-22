"""
Simple script to disintegrate ptn.pt into weights for table detection, 
structure recognition and cell detection.
"""

import torch
import os
import argparse
from termcolor import colored


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform inference on your table image."
    )
    parser.add_argument(
        "--pt-file",
        help="Path to downloaded weight (.pt).",
        required=True,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    weights_dict = torch.load(args.pt_file)

    table_det = weights_dict["table_detector"]
    structure_rec = weights_dict["structure_recognizer"]
    cell_det = weights_dict["cell_detector"]

    os.makedirs("weights/", exist_ok=True)

    # Save separate weights
    torch.save(table_det, "weights/table_det.pth")
    print(
        colored(
            "Saved table_det.pth.",
            "cyan",
        )
    )
    torch.save(structure_rec, "weights/structure_rec.pth")
    print(
        colored(
            "Saved structure_rec.pth.",
            "cyan",
        )
    )
    torch.save(cell_det, "weights/cell_det.pth")
    print(
        colored(
            "Saved cell_det.pth.",
            "cyan",
        )
    )
