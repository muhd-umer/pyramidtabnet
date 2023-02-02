# PyramidTabNet

## An End-to-End Approach to Table Analysis in Scanned Documents

Transformers have been making great strides in the field of computer vision in recent years, and their impact has been particularly noteworthy in the domain of document analysis. In this paper, we introduce PyramidTabNet (PTN), a method that builds upon the performance of the convolution-less Pyramid Vision Transformer by incorporating a structural modification data augmentation technique before training the architecture. Specifically, the augmentation process consists of three sequential pipelines, namely, clustering, fusion, and patching, for generation of new document images as well as for masking text. Our proposed pipeline demonstrates significant performance improvements over other augmentation techniques and surpasses recent works in table detection metrics. Additionally, it achieves performance comparable to the current state-of-the-art methods for structure recognition tasks.


## Dependencies
_It is recommended to create a new virtual environment so that updates/downgrades of packages do not break other projects._

- Environment characteristics
  <br/>`python = 3.9.12` `torch = 1.11.0` `cuda = 11.3` `torchvision = 0.12.0`
- This repo uses toolboxes provided by `OpenMMLab` to train and test models. Head over to the official documentation of [MMDetection](https://github.com/open-mmlab/mmdetection) for [installation instructions](https://mmdetection.readthedocs.io/en/latest/) if you want to train your own model.
- Alternatively, if all you want to do is to test the model, you can install `mmdet` as a third-party package. Run:
```python
pip install -r requirements.txt
```
- After all the packages has been successfully installed, install `mmcv` by executing the following commands:
```python
pip install -U openmim
mim install mmcv-full==1.6.0
```
## Datasets

We provide the test set of `cTDaR - TRACK A` in `COCO JSON format` by default (for evaluation). You can access the full cTDaR dataset from the following publicly available GitHub repo: [cTDaR - All Tracks](https://github.com/cndplab-founder/ICDAR2019_cTDaR). Other public datasets can be downloaded and placed in [data](data/) directory for training/evaluation.

## Data Augmentation

Refer to [augmentation](augmentation#data-augmentation) directory for instructions on how to use the scripts to generate new document images.

## Run <a href="https://colab.research.google.com/github/muhd-umer/pyramidtabnet/blob/main/resources/pyramidtabnet.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>

Following sections provide instructions to evaluate and/or train PyramidTabNet on your own data.<br/>
_Note: It is recommended to execute the scripts from the project root in order to utilize the relative paths to the test set._

### Training

- Refer to [Data Augmentation](augmentation) to generate additional training samples to improve model performance. ❤️
- Before firing up the `train.py` script, make sure to configure the data keys in the config file
- _Refer to [MMDetection documentation](https://mmdetection.readthedocs.io/en/latest/2_new_data_model.html#train-with-customized-datasets) for more details on how to modify the keys._

```python
python model/train.py path/to/config/file --gpu-id 0
```

- Alternatively, you can launch training on multiple GPUs using the following script:

```powershell
bash model/dist_train.sh ${CONFIG_FILE} \
                         ${GPU_NUM} \
                         [optional args]
```

### Evaluation

- Download link of fine-tuned weights are available in [this section.](#weights--metrics).
- Execute `test.py` with the appropriate command line arguments. Example usage:

```python
python model/test.py --config-file path/to/config/file \
                     --input path/to/directory \
                     --weights path/to/finetuned/checkpoint \
                     --device "cuda"
```

### Inference

- To perform end-to-end table analysis (visualize detections) on a `single image/test directory`, execute `main.py`. Download the weights from [Weights & Metrics](#weights--metrics) and place them in the [weights/](weights/) directory. Example usage:

```python
python main.py --config-file path/to/config/file \
               --input path/to/input/image or directory \
               --weights-dir path/to/weights/directory \
               --device "cuda"
```

### Detection Inference

- To perform table detection on a `single image/test directory`, execute `td.py`. Example usage:

```python
python model/td.py --config-file path/to/config/file \
                   --input path/to/input/image or directory \
                   --weights path/to/detection/weights \
                   --device "cuda" \
                   --save
```

### Recognition Inference

- To perform stucture recognition on a `single image/test directory`, execute `tsr.py`. Example usage:

```python
python model/tsr.py --config-file path/to/config/file \
                    --input path/to/input/image or directory \
                    --structure-weights path/to/structure/weights \
                    --cell-weights path/to/cell/weights \
                    --device "cuda" \
                    --save
```

## Weights & Metrics

Evaluation metrics are displayed in the following tables. Note: End-user should place the downloaded weights in the [weights/](weights/) directory for a streamlined evaluation of scripts.

- To download all the weights, execute:
```powershell
bash weights/get_weights.sh
bash weights/fine_tuned.sh
```
<div align="center">

**Table Detection**

| <div align="center">Model</div> | <div align="center">Dataset</div> | <div align="center">Precision</div> | <div align="center">Recall</div> | <div align="center">F1</div> | <div align="center">Link</div> |
| --- | --- | --- | --- | --- | --- |
| PyramidTabNet | ICDAR 2017-POD <br> ICDAR 2019 <br> UNLV <br> Marmot <br> TableBank <br> | 99.8 <br> - <br> 97.7 <br> 92.1 <br> 98.9 | 99.3 <br> - <br> 94.9 <br> 98.2 <br> 98.2 | 99.5 <br> 98.7 <br> 96.3 <br> 95.1 <br> 98.5 | [Link](https://github.com/muhd-umer/pyramidtabnet/releases/download/v0.1.0/icdar2017.pth) <br> [Link](https://github.com/muhd-umer/pyramidtabnet/releases/download/v0.1.0/icdar2019.pth) <br> [Link](https://github.com/muhd-umer/pyramidtabnet/releases/download/v0.1.0/unlv.pth) <br> [Link](https://github.com/muhd-umer/pyramidtabnet/releases/download/v0.1.0/marmot.pth) <br> [Link](https://github.com/muhd-umer/pyramidtabnet/releases/download/v0.1.0/tablebank.pth) |

</div>

<div align="center">

**Table Structure Recognition**

| <div align="center">Model</div> | <div align="center">Dataset</div> | <div align="center">Precision</div> | <div align="center">Recall</div> | <div align="center">F1</div> |
| --- | --- | --- | --- | --- |
| PyramidTabNet | ICDAR 2013 <br> SciTSR <br> FinTabNet <br>| 92.3 <br> 98.4 <br> 93.2 | 95.3 <br> 99.1 <br> 88.6 | 93.8 <br> 98.7 <br> 90.8|

</div> 

_Note: FinTabNet fine-tuned model is for cell-detection._

<div align="center">

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

Resolve it by passing in the correct command line argument for `--config-file`.

## Acknowledgements

**Special thanks to the following contributors without which this repo would not be possible:**

1. The [MMDetection](https://github.com/open-mmlab/mmdetection) team for creating their amazing framework to push the state of the art computer vision research and enabling us to experiment and build various models very easily.
<p align="center">
   <a href="https://github.com/open-mmlab/mmdetection"><img width="220" height="75" src="https://raw.githubusercontent.com/open-mmlab/mmdetection/master/resources/mmdet-logo.png"/></a>
</p>

2. The authors of [Pyramid Vision Transformer (PVT v2)](https://arxiv.org/pdf/2106.13797.pdf) for their wonderful contribution to enhance advancements in computer vision.

3. The authors of [Craft Text Detector](https://arxiv.org/abs/1904.01941) for their awesome repository for text detection.

4. The author of [mAP Repo](https://github.com/Cartucho/mAP) for providing a straightforward script to evaluate deep learning models for object detection metrics.

5. [Google Colaboratory](https://github.com/googlecolab) for providing free-high end GPU resources for research and development. All of the code base was developed using their platform and could not be possible without it.
