# PyramidTabNet
## An End-to-End Approach to Table Analysis in Scanned Documents
Transformers have been making great strides in the field of computer vision in recent years, and their impact has been particularly noteworthy in the domain of document analysis. In this paper, we introduce PyramidTabNet (PTN), a method that builds upon the performance of the convolution-less Pyramid Vision Transformer (PVT v2) by incorporating a structural modification data augmentation technique before training the architecture. Specifically, the augmentation process consists of three sequential pipelines, namely, clustering, fusion, and patching, for generation of new document images as well as for masking text. Notably, our proposed pipeline surpasses other augmentation strategies on all fronts, and achieves comparable or better results than recent works on various publicly available table analysis datasets.

## Dependencies
*It is recommended to create a new virtual environment so that updates/downgrades of packages do not break other projects.*
- Environment characteristics
<br/>`python = 3.9.12` `torch = 1.10.0` `cuda = 11.3` `torchvision = 0.11.0`
- This repo uses toolboxes provided by `OpenMMLab` to train and test models. Head over to the official documentation of [MMDetection](https://github.com/open-mmlab/mmdetection) for [installation instructions](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation).

```
pip install -r requirements.txt
```

## Datasets
- Table Detection - We provide the test set of `cTDaR - TRACK A` in `COCO JSON format` by default (for evaluation purposes). You can access the full cTDaR dataset from the following publicly available GitHub repo: [cTDaR - All Tracks](https://github.com/cndplab-founder/ICDAR2019_cTDaR)

- Table Structure Recognition - You can access download links to [FinTabNet](https://developer.ibm.com/exchanges/data/all/fintabnet/) from the official IBM developer website. We also provide ICDAR2013 test set by default.

## Data Augmentation
Refer to [augmentation](augmentation/) directory for instructions on how to use the scripts to generate new document images.

## Run
Following sections provide instructions to evaluate and/or train PyramidTabNet on your own data.<br/>
*Note: It is recommended to execute the scripts in this directory from the project root in order to utilize the relative paths to the test set.*
### Evaluation
- Download link of fine-tuned weights are available in [this table](https://github.com/muhd-umer/PyramidTabNet#table-detection).
- Execute `test.py` with the appropriate command line arguments. Example usage:
```python
python model/test.py --config-file path/to/config/file \
                     --det-weights path/to/finetuned/checkpoint \
                     --data-dir data/cTDaR/ \
                     --device "cuda"
```

### End-to-end inference
- To perform end-to-end table analysis (visualize detections/extract bounding box coordinates of tables) on a single image, execute `run.py`. Download the weights from [Weights & Metrics](#weights--metrics) and place them in the [weights/](weights/) directory. Example usage:
```python
python run.py --config-file path/to/config/file \
              --weights-dir path/to/weights/dir \
              --input-img path/to/input/image \
              --device "cuda"
```

### Detection Inference
- To perform either detection or stucture recognition on a single image, execute `inference.py`. Example usage:
```python
python model/inference.py --config-file path/to/config/file \
                          --input-img path/to/input/image \
                          --weights path/to/finetuned/checkpoint \
                          --device "cuda"
```

### Training
- Refer to [Data Augmentation](https://github.com/muhd-umer/PyramidTabNet/tree/main/detection/augmentation) to generate additional training samples to improve model performance. ❤️
- Before firing up the `train.py` script, make sure to configure the data keys in the config file 
- *Refer to [MMDetection documentation](https://mmdetection.readthedocs.io/en/latest/2_new_data_model.html#train-with-customized-datasets) for more details on how to modify the keys.*
```python
python train.py path/to/config/file --gpu-id 0
```
*Note: A distributed training script is not bundled with this repo, however, you can refer to the official MMDetection repo for one.*

## Weights & Metrics
### Table Detection
The results of table detection on `ICDAR 2019 cTDaR` are shown below. The weights (.pth) file are embedded into the model column of the following table.

<div align="center">

| Model | Weighted F1 | IoU<sup>@.6</sup> | IoU<sup>@.7</sup> | IoU<sup>@.8</sup> | IoU<sup>@.9</sup> |
|:---:|:---:|:---:|:---:|:---:|:---:|
| DeiT-B | 93.07 | 95.51 | 94.61 | 93.48 | 89.89 |
| BEiT-B | 94.25 | 96.06 | 95.39 | 95.16 | 91.34 |
| MAE-B | 93.81 | 96.47 | 95.58 | 94.48 | 90.07 |
| DiT-B | 94.74 | 96.29 | 95.61 | 95.39 | 92.46 |
| DiT-L | 95.50 | 98.00 | 97.56 | 96.23 | 91.57 |
| DiT-B (Cascade) | 95.85 | 97.33 | 96.89 | 96.67 | 93.33 |
| DiT-L (Cascade) | 96.29 | 97.89 | 97.22 | 97.00 | 93.88 |
| [PyramidTabNet](https://drive.google.com/file/d/1DN_DSM-wb5izSoL7PkBirL3_R7y-tK1i/view?usp=share_link) | 97.02 | 98.45 | 98.45 | 97.57 | 94.47 |

</div>

### Table Structure Recognition
The results of table detection on `ICDAR 2013` are shown below. The weights (.pth) file are embedded into the model column of the following table.

<div align="center">

| Model | Precision | Recall | F1 |
|:---:|:---:|:---:|:---:|
| DeepDeSRT | 95.91 | 87.42 | 91.44 |
| SPLERGE | 91.22 | 91.14 | 91.92 |
| BI-directional GRU | 96.93 | 90.14 | 93.42 |
| TabStructNet | 93.01 | 90.81 | 91.92 |
| [PyramidTabNet](https://drive.google.com/file/d/1v1ndhJlgmEtvgTxrlpCE9jycNEAiehVN/view?usp=share_link) | 93.53 | 90.74 | 92.11 |
</div>

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
