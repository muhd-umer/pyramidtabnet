# Table Detection
<div align="center">
  <img src="https://github.com/muhd-umer/PyramidTabNet/blob/main/resources/detections.png" width="1000"/>
  <p align="center">Model outputs on ICDAR 2019 cTDaR test set samples</p1>
</div>

## Data Preparation
- To download dataset for table detection, refer to [Datasets](https://github.com/muhd-umer/PyramidTabNet#datasets). ❤️
- For evaluation purposes, we provide the `COCO JSON` annotations as well as the original images of the cTDaR test set, so you can run the [evaluation](https://github.com/muhd-umer/PyramidTabNet/edit/main/detection/README.md#evaluation) section rifght after downloading the weights.

## Run
Following sections provide instructions to evaluate and/or train PyramidTabNet on your own data.<br/>
*Note: It is recommended to execute the scripts in this directory from the project root in order to utilize the relative paths to the test set.*
### Evaluation
- Download link of fine-tuned weights are available in [this table](https://github.com/muhd-umer/PyramidTabNet#table-detection).
- Execute `test.py` with the appropriate command line arguments. Example usage:
```python
python detection/test.py --config-file path/to/config/file \
                         --det-weights path/to/finetunes/checkpoint \
                         --data-dir data/cTDaR/ \
                         --device "cuda"
```

### Inference
- To perform inference (visualize detections/extract bounding box coordinates of tables) on a single image, execute `inference.py`.
- Example usage:
```python
python detection/inference.py --config-file path/to/config/file \
                              --input-img path/to/input/image
                              --det-weights path/to/finetunes/checkpoint \
                              --device "cuda"
```

### Training
- Refer to [Data Augmentation](https://github.com/muhd-umer/PyramidTabNet/tree/main/detection/augmentation) to generate additional training samples to improve model performance. ❤️
- Before firing up the `train.py` script, make sure to configure the data keys in the config file 
- *Refer to [MMDetection documentation](https://mmdetection.readthedocs.io/en/latest/2_new_data_model.html#train-with-customized-datasets) for more details on how to modify the keys.*
```python
python detection/train.py path/to/config/file --gpu-id 0
```
*Note: A distributed training script is not bundled with this repo, however, you can refer to the official MMDetection repo for one.*
## Acknowledgements
The [MMDetection](https://github.com/open-mmlab/mmdetection) team for creating their amazing framework to push the state of the art computer vision research and enabling us to experiment and build various models very easily.
<p align="center">
   <a href="https://github.com/open-mmlab/mmdetection"><img width="235" height="75" src="https://raw.githubusercontent.com/open-mmlab/mmdetection/master/resources/mmdet-logo.png"/></a>
</p>
