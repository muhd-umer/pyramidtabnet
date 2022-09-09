"""
Copyright (c) OpenMMLab. All rights reserved.

Modified by Muhammad Umer, Author of PyramidTabNet

Modified single_gpu_test() to support cTDaR metrics.
Reference: https://github.com/open-mmlab/mmdetection/blob/master/tools/test.py
"""

import os.path as osp
import torch

import mmcv
from mmcv.image import tensor2imgs
from mmdet.core import encode_mask_results


def evaluate(model, data_loader, show=False, out_dir=None, show_score_thr=0.8):
    model.eval()
    results_dict = {}
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, "PALETTE", None)
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            img_filename = data["img_metas"][0].data[0][0]["ori_filename"]
            results_dict[img_filename] = [bbox_results for bbox_results, _ in result][0]

        batch_size = len(result)
        if show:
            if batch_size == 1 and isinstance(data["img"][0], torch.Tensor):
                img_tensor = data["img"][0]
            else:
                img_tensor = data["img"][0].data[0]
            img_metas = data["img_metas"][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]["img_norm_cfg"])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta["img_shape"]
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta["ori_shape"][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta["ori_filename"])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr,
                )

        # encode mask results
        if isinstance(result[0], tuple):
            result = [
                (bbox_results, encode_mask_results(mask_results))
                for bbox_results, mask_results in result
            ]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and "ins_results" in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]["ins_results"]
                result[j]["ins_results"] = (
                    bbox_results,
                    encode_mask_results(mask_results),
                )

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()

    return results_dict, results
