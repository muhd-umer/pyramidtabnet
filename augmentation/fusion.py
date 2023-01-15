"""
MIT License:
Copyright (c) 2022 Muhammad Umer

Data generator script for fused 
    tables after forming clusters

Following rules are observed:
Rules of patching:
    Horizontal contours:
    If post-cutoff height of table is > than 700,
        or < 250, do not write it.

    Vertical contours:
    If post-cutoff width of table is > 1200,
        or < 300, do not write it.
"""

from bbox_utils import *
import os
import os.path as osp
import argparse
from tqdm import tqdm
from termcolor import colored
from pascal_voc_writer import Writer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate your table detector.")
    parser.add_argument(
        "--input-dir",
        help="Path to files for generating new tables.)",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        help="Path to output where fused tables are saved.",
        default="joined/",
        required=False,
    )
    parser.add_argument(
        "--type",
        help="Contour detection type. (horizontal/vertical). i.e. Horizontal contours will join tables in a vertical manner.",
        default="horizontal",
        required=False,
    )
    parser.add_argument(
        "--num-samples",
        help="Number of samples to generate/join. We recommend keeping it 1/3rd the size of input-dir.",
        default=50,
        required=False,
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    imgdir = sorted(os.listdir(args.input_dir))
    assert len(imgdir) > int(args.num_samples), colored(
        "Input directory must be larger than the desired number of samples. "
        "We recommend setting num_samples to 1/3rd the size of input directory.",
        "red",
    )
    skip = list()
    count = 0

    for y in tqdm(
        range(int(args.num_samples)),
        colour="cyan",
        desc=colored("Generating Images", "cyan"),
    ):
        """
        Chooses a random batch (n = 2) to join
        after detecting contours and splitting
        """
        for k in range(len(imgdir)):
            if not y == k:
                if imgdir[k] not in skip:
                    base_name = "cTDaR_t1" + imgdir[y][9:-6] + imgdir[k][9:-6] + ".png"

                    image_a = cv2.imread(
                        osp.join(args.input_dir, imgdir[y]), cv2.IMREAD_COLOR
                    )
                    image_b = cv2.imread(
                        osp.join(args.input_dir, imgdir[k]), cv2.IMREAD_COLOR
                    )
                    """
                    If the tables are two different in size, skip
                    """
                    if args.type == "horizontal":
                        ksize = (15, 1)
                        image_b = resize_to_width(image_a, image_b)
                        if image_b.shape == (1, 1):
                            continue
                    else:
                        ksize = (1, 15)
                        image_b = resize_to_height(image_a, image_b)
                        if image_b.shape == (1, 1):
                            continue

                    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
                    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

                    thresh_a = cv2.threshold(
                        gray_a, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )[1]
                    thresh_b = cv2.threshold(
                        gray_b, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )[1]
                    cutoff_a = get_cutoff(thresh_a, ksize)
                    cutoff_b = get_cutoff(thresh_b, ksize)

                    """
                    Fusion conditions, follows the Rules defined at the beginning
                        of this file
                    """
                    if args.type == "horizontal":
                        join_a = image_a[0:cutoff_a, 0 : image_a.shape[1]]
                        join_b = image_b[0:cutoff_b, 0 : image_b.shape[1]]
                        if join_a.shape[0] < 100:
                            join_a = image_a[
                                cutoff_a : image_a.shape[0], 0 : image_a.shape[1]
                            ]
                        if join_b.shape[0] < 100:
                            join_b = image_b[
                                cutoff_b : image_b.shape[0], 0 : image_b.shape[1]
                            ]
                        joined = cv2.vconcat([join_b, join_a])
                        if not (joined.shape[0] > 700 or joined.shape[0] < 250):
                            cv2.imwrite(osp.join(args.output_dir, base_name), joined)
                            count += 1
                            skip.append(imgdir[k])
                            break
                    else:
                        join_a = image_a[0 : image_a.shape[0], 0:cutoff_a]
                        join_b = image_b[0 : image_b.shape[0], 0:cutoff_b]
                        if join_a.shape[0] < 150:
                            join_a = image_a[
                                cutoff_a : image_a.shape[0], 0 : image_a.shape[1]
                            ]
                        if join_b.shape[0] < 150:
                            join_b = image_b[
                                cutoff_b : image_b.shape[0], 0 : image_b.shape[1]
                            ]
                        joined = cv2.hconcat([join_b, join_a])
                        if not (joined.shape[1] > 1200 or joined.shape[1] < 300):
                            cv2.imwrite(osp.join(args.output_dir, base_name), joined)
                            count += 1
                            skip.append(imgdir[k])
                            break

        if count == (int(args.num_samples) + 1):
            break


if __name__ == "__main__":
    main()
