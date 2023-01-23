# PyramidTabNet

## An End-to-End Approach to Table Analysis in Scanned Documents

Transformers have been making great strides in the field of computer vision in recent years, and their impact has been particularly noteworthy in the domain of document analysis. In this paper, we introduce PyramidTabNet (PTN), a method that builds upon the performance of the convolution-less Pyramid Vision Transformer by incorporating a structural modification data augmentation technique before training the architecture. Specifically, the augmentation process consists of three sequential pipelines, namely, clustering, fusion, and patching, for generation of new document images as well as for masking text. Our proposed pipeline demonstrates significant performance improvements over other augmentation techniques and surpasses recent works in table detection metrics. Additionally, it achieves performance comparable to the current state-of-the-art methods for structure recognition tasks.

## Dependencies

_It is recommended to create a new virtual environment so that updates/downgrades of packages do not break other projects._

- Environment characteristics
  <br/>`python = 3.9.12` `torch = 1.11.0` `cuda = 11.3` `torchvision = 0.12.0`
- This repo uses toolboxes provided by `OpenMMLab` to train and test models. Head over to the official documentation of [MMDetection](https://github.com/open-mmlab/mmdetection) for [installation instructions](https://mmdetection.readthedocs.io/en/latest/).

```python
pip install -r requirements.txt
```

## Datasets

We provide the test set of `cTDaR - TRACK A` in `COCO JSON format` by default (for evaluation). You can access the full cTDaR dataset from the following publicly available GitHub repo: [cTDaR - All Tracks](https://github.com/cndplab-founder/ICDAR2019_cTDaR). Other public datasets can be downloaded and placed in [data](data/) directory for training/evaluation.

## Data Augmentation

Refer to [augmentation](augmentation#data-augmentation) directory for instructions on how to use the scripts to generate new document images.

## Run

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
                   --save-detections
```

### Recognition Inference

- To perform stucture recognition on a `single image/test directory`, execute `tsr.py`. Example usage:

```python
python model/tsr.py --config-file path/to/config/file \
                    --input path/to/input/image or directory \
                    --structure-weights path/to/structure/weights \
                    --cell-weights path/to/cell/weights \
                    --device "cuda" \
                    --save-detections
```

## Weights & Metrics

Weights, along with evaluation metrics, are linked in this section. Note: End-user should place the downloaded weights in the [weights/](weights/) directory for a streamlined evaluation of scripts.

- Download the table detection models from the following table:
<div align="center">

### **Table Detection**

| <div align="center">Model</div> | <div align="center">Dataset</div> | <div align="center">Precision</div> | <div align="center">Recall</div> | <div align="center">F1</div> | <div align="center">Link</div> |
| --- | --- | --- | --- | --- | --- |
| PyramidTabNet | ICDAR 2017-POD <br> ICDAR 2019 <br> UNLV <br> Marmot <br> PubLayNet <br> | - <br> - <br> - <br> - <br> - | - <br> - <br> - <br> - <br> - | - <br> - <br> - <br> - <br> - | [Link]() <br> [Link]() <br> [Link]() <br> [Link]() <br> [Link]() |

</div>

- Download the table structure recognition models from the following table:
<div align="center">

### **Table Structure Recognition**

| <div align="center">Model</div> | <div align="center">Dataset</div> | <div align="center">Precision</div> | <div align="center">Recall</div> | <div align="center">F1</div> | <div align="center">Link</div> |
| --- | --- | --- | --- | --- | --- |
| PyramidTabNet | ICDAR 2013 <br> SciTSR <br> FinTabNet <br>| - <br> - <br> - <br> | - <br> - <br> - <br>| - <br> - <br> - <br>| [Link]() <br> - <br> [Link]() |

</div>

_Note: FinTabNet fine-tuned model is for cell-detection._

<div align="center">

**Download the concatenated best weights file (.pt) from [this link]().**

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

4. [Google Colaboratory](https://github.com/googlecolab) for providing free-high end GPU resources for research and development. All of the code base was developed using their platform and could not be possible without it.
