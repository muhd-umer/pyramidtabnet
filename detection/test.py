"""
Copyright (c) OpenMMLab. All rights reserved.

Modified by Muhammad Umer, Author of PyramidTabNet

Evaluation script for the table detector. Removed most, if not all, of distri-
buted inference capibilities as the test dataset is very small.
Reference: https://github.com/open-mmlab/mmdetection/blob/master/tools/test.py
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import os
import os.path as osp
from termcolor import colored

import mmcv
import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import get_dist_info, load_checkpoint, wrap_fp16_model
from utils import generate_xml, evaluate_table

from evaluation import evaluate
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.utils import (
    build_dp,
    compat_cfg,
    replace_cfg_vals,
    setup_multi_processes,
    update_data_root,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate your table detector.")
    parser.add_argument(
        "--config-file",
        help="Path to detector config file. (In configs/ directories.)",
        required=True,
    )
    parser.add_argument(
        "--det-weights",
        help="Checkpoint file to load weights from.",
        required=True,
    )
    parser.add_argument(
        "--data-dir", type=str, help="Path to test cTDaR dataset.", required=True
    )
    parser.add_argument(
        "--output-dir",
        help="Path to output where picke file as well as metrics are saved.",
        default="output/",
        required=False,
    )
    parser.add_argument(
        "--device",
        help="Choose inference runtime, either CPU or CUDA. (Defaults to CPU)",
        default="cuda",
        required=False,
    )
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse Conv and BN; Results in a slight increase in inference speed.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help="Evaluation metrics; depend on your dataset. "
        "'bbox', 'segm' for COCO Dataset",
        required=False,
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Saves the predictions plotted on test images in the --output-dir.",
    )
    parser.add_argument(
        "--coco-save",
        action="store_true",
        help="Saves COCO evaluation metrics in a .json file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Detection Threshold (Default: 0.7)",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


if __name__ == "__main__":
    args = parse_args()

    ROOT_DIR = osp.abspath(args.data_dir).replace(os.sep, "/")
    included_extensions = ["jpg"]
    test_filenames = [
        fn
        for fn in os.listdir(osp.join(args.data_dir, "test"))
        if any(fn.endswith(ext) for ext in included_extensions)
    ]
    picke_dir = osp.join(args.output_dir, "results.pkl")

    cfg = Config.fromfile(args.config_file)
    cfg = replace_cfg_vals(cfg)

    update_data_root(cfg)

    # Update some keys in accordance with arguments
    cfg.data.test.ann_file = osp.join(args.data_dir, "test.json")
    cfg.data.test.img_prefix = osp.join(args.data_dir, "test/")

    if args.device == "cuda":
        assert torch.cuda.is_available(), f"No CUDA Runtime found."
    elif args.device == "cpu":
        print(
            colored(
                "Device = 'cpu' -> slow_conv2d_cpu is not implemented for Half tensors; Setting cfg.fp16 to None.",
                "red",
            )
        )
        cfg.fp16 = None

    cfg.device = args.device

    cfg = compat_cfg(cfg)

    # Set multi-process settings
    setup_multi_processes(cfg)

    # Set CUDNN_Benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    if "pretrained" in cfg.model:
        cfg.model.pretrained = None

    elif "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get("neck"):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get("rfp_backbone"):
                    if neck_cfg.rfp_backbone.get("pretrained"):
                        neck_cfg.rfp_backbone.pretrained = None

        elif cfg.model.neck.get("rfp_backbone"):
            if cfg.model.neck.rfp_backbone.get("pretrained"):
                cfg.model.neck.rfp_backbone.pretrained = None

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False
    )

    # In case the dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get("samples_per_gpu", 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get("samples_per_gpu", 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get("test_dataloader", {}),
    }

    rank, _ = get_dist_info()

    if args.output_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.output_dir))
        json_file = osp.join(args.output_dir, f"coco_metrics.json")

    # Build the PyTorch Dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # Build the detector and load the checkpoint weights.
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))

    # Use half tensor which preserve memory, and hence is preferred.
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # Load weights into the config model.
    checkpoint = load_checkpoint(model, args.det_weights, map_location="cpu")

    # Fuse Conv and BN to increase inference speed.
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    # For Backward compatibility; save class info in checkpoints
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    # Build DataParallel module and do testing on a single GPU.
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    outputs_dict, outputs = evaluate(
        model, data_loader, args.save_images, args.output_dir, args.threshold
    )

    xml_dir = generate_xml(
        args.output_dir, test_filenames, outputs_dict, args.threshold
    )

    # Pretty prints a table with cTDaR 2019 style metrics
    wF1 = evaluate_table(xml_dir)

    rank, _ = get_dist_info()

    if rank == 0:
        if args.output_dir:
            print(f"\nWriting results to {picke_dir}.")
            mmcv.dump(outputs, picke_dir)

        kwargs = {}
        if args.eval and args.coco_save:
            eval_kwargs = cfg.get("evaluation", {}).copy()

            # Remove EvalHook args
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
                "dynamic_intervals",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            metric = dataset.evaluate(outputs, **eval_kwargs)
            metric_dict = dict(config=args.config_file, metric=metric)

            if args.output_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)
