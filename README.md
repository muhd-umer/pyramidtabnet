# PyramidTabNet
## An End-to-End Approach to Table Analysis in Scanned Documents
Transformers have presented encouraging progress in the domain of computer vision for the past several years. Our proposed approach is based on the improved baselines of convolution-less Pyramid Vision Transformer (PVT v2) paired with novel data augmentation techniques in the field of document analysis. Notably, our proposed pipeline achieves comparable or better results on various publicly available table analysis datasets than recent works such as Document Image Transformer (DiT).

## Dependencies
*It is recommended to create a new virtual environment so that updates/downgrades of packages do not break other projects.*
- Environment characteristics
<br/>`python = 3.9.12` `torch = 1.10.0` `cuda = 11.3` `torchvision = 0.11.0`
- This repo uses toolboxes provided by `OpenMMLab` to train and test models. Head over to the official documentation of [MMDetection](https://github.com/open-mmlab/mmdetection) for [installation instructions](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation).

```
pip install -r requirements.txt
```

## Installation / Run
To get started, clone this repo and install the required dependencies.

### Datasets
- **Table Detection** - We provide the test set of `cTDaR - TRACK A` in `COCO JSON format` by default (for evaluation purposes). You can access the full cTDaR dataset from the following publicly available GitHub repo: [cTDaR - All Tracks](https://github.com/cndplab-founder/ICDAR2019_cTDaR)

- **Table Structure Recognition** - You can access download links to [FinTabNet](https://developer.ibm.com/exchanges/data/all/fintabnet/) from the official IBM developer website.

### Table Detection
The results of table detection on ICDAR 2019 cTDaR are shown below. The instructions to reproduce the results can be found inside [PyramidTabNet/detection](detection/README.md).

| Model |  Weighted F1 | IoU<sup>@.6</sup> | IoU<sup>@.7</sup> | IoU<sup>@.8</sup> | IoU<sup>@.9</sup> | Checkpoint |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PyramidTabNet | 97.02 | 98.45 | 98.45 | 97.57 | 94.47 | [Weights (.pth)](https://drive.google.com/file/d/1DN_DSM-wb5izSoL7PkBirL3_R7y-tK1i/view?usp=share_link) |

### Table Structure Recognition
Evaluation summary on `cTDaR - TRACK A Modern` is presented below. We also provide the fine-tuned weights. The instructions to reproduce the results can be found at recognition/README.md.

Model |  Weighted F1 | | | | | Checkpoint |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PyramidTabNet | | | | | | |

## Common Issues
- Machines running variants of Microsoft Windows encounter issues with mmcv imports. Follow the [installation guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) on the official MMCV documentation to resolve such issues. Example:

```TypeScript
ModuleNotFoundError: No module named 'mmcv._ext'
```

- For table detection, if you get an error of the following form:

```TypeScript
Error(s) in loading state_dict for TDModel; Missing key(s) in state_dict
```
Resolve it by passing in the correct command line argument for `--config-file`. All configs for table detection are present in `detection/configs`.

## Acknowledgements
**Special thanks to the following contributors without which this repo would not be possible:**
1. The [MMDetection](https://github.com/open-mmlab/mmdetection) team for creating their amazing framework to push the state of the art computer vision research and enabling us to experiment and build various models very easily.
<p align="center">
   <a href="https://github.com/open-mmlab/mmdetection"><img width="220" height="75" src="https://raw.githubusercontent.com/open-mmlab/mmdetection/master/resources/mmdet-logo.png"/></a>
</p>

2. The authors of [Pyramid Vision Transformer (PVT v2)](https://arxiv.org/pdf/2106.13797.pdf) for their wonderful contribution to enhance advancements in computer vision.

3. [Google Colaboratory](https://github.com/googlecolab) for providing free high end GPU resources for research and development. All of the code base was developed using their platform and could not be possible without it.
