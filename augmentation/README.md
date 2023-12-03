# Data Augmentation
This directory contains data augmentation scripts for PyramidTabNet. An input directory is first clustered to reduce variancy in data augmentation. This process is followed by fusion of tables within the clusters to create new table images. Lastly, we patch the tables generated onto the training set documents to pass onto the model for training.

## Scripts to Generate Augmented Data

### K-Means Clustering
- We cluster the input images to patch (from the training set) as well as table images (to patch onto the training images) collected from external sources as they make patching more consistent and reduces variancy, *i.e. a document without any colored tables may get patched with a colored one, leading to undesired learnings by the model.*

- You can cluster data by executing `clusters.py`. Example usage:

    ```python
    python augmentation/clusters.py --f path/to/images/to/cluster \
                                    --k number-of-clusters \  # must be an integer
                                    --m  # move instead of copying
    ```
    
### Fusion
Complex table images are fused together to form new tables to be fed into the patching pipeline. Following steps are employed in order to fuse two table images from the same cluster:
- Find the horizontal/vertical contours on a random batch (i.e., n=2) of tables from the input directory and find a cutoff point. This point is used to split the tab into two vertically/horizontally.
- Concatenate the two images to generate a new one.
You can generate mix of tables by executing fusion.py. Example usage:

    ```python
    python augmentation/fusion.py --input-dir path/to/training/images \
                                --output-dir path/to/generated/data \
                                --num-samples integer-value
                                # for optimum results, 
                                # num-samples = 1/3 * len(input-dir)
    ```

### Patching
Patching of tables to input images can be characterized by the following pipeline:
- Find the largest area on the input image that isn't already occupied by another table. Largest area ensures that we mask our patches over as much of text/figures as possible.
- Pad the table images to the width of the image.
- Paste the table at a proper scale on the center of the largest area.
- Generate output masks and PASCAL-VOC annotations.
You can generate augmented data by executing patcher.py. Example usage:

    ```python
    python augmentation/patcher.py --input-dir path/to/training/images \
                                --input-masks path/to/training/masks \
                                --patch-dir path/to/tables/to/patch \
                                --output-dir path/to/generated/data
    ```
