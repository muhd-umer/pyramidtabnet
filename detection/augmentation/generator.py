"""
MIT License:
Copyright (c) 2022 Muhammad Umer

Data generator script for in-patching and
external patching of tables to generate
            new data
"""

from turtle import color
from utils import *
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
        help="Path to files on which you want to patch tables.)",
        required=True,
    )
    parser.add_argument(
        "--input-masks",
        help="Path to input masks to ensure that tables dont overlap.)",
        required=True,
    )
    parser.add_argument(
        "--patch-dir",
        help="Path to external tables that you want to patch.",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        help="Path to output where picke file as well as metrics are saved.",
        default="generated/",
        required=False,
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(osp.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(osp.join(args.output_dir, "masks"), exist_ok=True)
    os.makedirs(osp.join(args.output_dir, "xml"), exist_ok=True)

    imgdir = sorted(os.listdir(args.input_dir))
    to_pastedir = sorted(os.listdir(args.patch_dir))

    skip = list()
    terminate = 0

    for y in tqdm(
        range(len(to_pastedir)),
        colour="cyan",
        desc=colored("Generating Patched (Augmented) Images", "cyan"),
    ):
        """
        to_pastedir contains the external tables to be patched onto
        the training images to mask text and create new images
        """
        count = 0
        for k in range(len(imgdir)):
            if imgdir[k] not in skip:
                terminate = 0
                count += 1

                img = cv2.imread(osp.join(args.input_dir, imgdir[k]), cv2.IMREAD_COLOR)
                _, imgwidth = img.shape[:2]
                mask = cv2.imread(
                    osp.join(args.input_masks, imgdir[k]), cv2.IMREAD_GRAYSCALE
                )

                paste = cv2.imread(
                    osp.join(args.patch_dir, to_pastedir[y]), cv2.IMREAD_COLOR
                )
                """
                Main Loop: Padding to remove text near the table borders
                """
                pad_paste = cv2.copyMakeBorder(
                    paste, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )
                pasteheight, pastewidth = pad_paste.shape[:2]

                copyimg = img.copy()
                copymask = mask.copy()
                ori_bbox = get_bbox(mask)

                mask = expand_mask(mask, ori_bbox)
                paste_location = get_paste_location(mask)
                loc_width = get_width(paste_location)
                loc_height = get_height(paste_location)
                centerX, centerY = get_center(paste_location)

                xmin, xmax = round(centerX - pastewidth / 2), round(
                    centerX + pastewidth / 2
                )
                ymin, ymax = round(centerY - pasteheight / 2), round(
                    centerY + pasteheight / 2
                )
                paste_bbox = [xmin, ymin, xmax, ymax]

                for i in range(len(ori_bbox)):
                    if get_overlap_status(ori_bbox[i], paste_bbox):
                        terminate = 1

                if (
                    loc_height > pasteheight + 10
                    and loc_width > pastewidth + 10
                    and terminate == 0
                ):
                    resized_paste = cv2.resize(
                        pad_paste, (xmax - xmin, ymax - ymin), cv2.INTER_AREA
                    )

                    copyimg[ymin:ymax, xmin:xmax] = resized_paste
                    copyimg[ymin:ymax, 0:xmin] = 255
                    copyimg[ymin:ymax, xmax:imgwidth] = 255
                    copymask[ymin + 30 : ymax - 30, xmin + 30 : xmax - 30] = 255

                    cv2.imwrite(
                        osp.join(
                            args.output_dir, "images", f"{y}_" + f"{k}_" + imgdir[k]
                        ),
                        copyimg,
                    )
                    cv2.imwrite(
                        osp.join(
                            args.output_dir, "masks", f"{y}_" + f"{k}_" + imgdir[k]
                        ),
                        copymask,
                    )
                    skip.append(imgdir[k])
                    break

        if count == (len(imgdir) - 1):
            """
            On the odd case that a table is passed that cant
            be patched onto training images due to their size,
            it is downscaled heavily to viably patch it
            """
            scale_percent = round(650 / pad_paste.shape[1], 2)
            width = int(pad_paste.shape[1] * scale_percent)
            height = int(pad_paste.shape[0] * scale_percent)
            dim = (width, height)
            pad_paste = cv2.resize(pad_paste, dim, interpolation=cv2.INTER_AREA)

            if loc_height > pasteheight + 10 and loc_width > pastewidth + 10:
                xmin, xmax = round(centerX - pastewidth / 2), round(
                    centerX + pastewidth / 2
                )
                ymin, ymax = round(centerY - pasteheight / 2), round(
                    centerY + pasteheight / 2
                )

                resized_paste = cv2.resize(
                    pad_paste, (xmax - xmin, ymax - ymin), cv2.INTER_AREA
                )

                copyimg[ymin:ymax, xmin:xmax] = resized_paste
                copyimg[ymin:ymax, 0:xmin] = 255
                copyimg[ymin:ymax, xmax:imgwidth] = 255
                copymask[ymin + 30 : ymax - 30, xmin + 30 : xmax - 30] = 255

                cv2.imwrite(
                    osp.join(args.output_dir, "images", f"{y}_" + f"{k}_" + imgdir[k]),
                    copyimg,
                )
                cv2.imwrite(
                    osp.join(args.output_dir, "masks", f"{y}_" + f"{k}_" + imgdir[k]),
                    copymask,
                )

    """
    Save PASCAL-VOC annotations of generated data
    """
    output_mask_list = os.listdir(osp.join(args.output_dir, "masks"))

    for i in tqdm(
        range(len(output_mask_list)),
        colour="magenta",
        desc=colored("Generating PASCAL-VOC Annotations", "magenta"),
    ):
        mask = cv2.imread(
            osp.join(args.output_dir, "masks", output_mask_list[i]),
            cv2.IMREAD_GRAYSCALE,
        )
        writer = Writer(
            osp.join(args.output_dir, "masks", output_mask_list[i]),
            mask.shape[1],
            mask.shape[0],
        )

        bbox = get_bbox(mask)

        try:
            for k in range(len(bbox)):
                writer.addObject(
                    "table", bbox[k][0], bbox[k][1], bbox[k][2], bbox[k][3]
                )

        except TypeError:
            os.remove(osp.join(args.output_dir, "images", output_mask_list[i]))
            os.remove(osp.join(args.output_dir, "masks", output_mask_list[i]))
            """
            Remove corrupt/ tableless images
            """

        # write to file
        writer.save(osp.join(args.output_dir, "xml", output_mask_list[i][:-4] + ".xml"))

    print(colored(f"Saved Augmented Data at {args.output_dir}", "green"))


if __name__ == "__main__":
    main()
