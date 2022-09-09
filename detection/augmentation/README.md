# Data Augmentation
## Scripts to Generate Augmented Data

### K-Means Clustering
- We cluster the input images to patch (from the training set) as well as table images (to patch onto the training images) collected from external sources as they make patching more consistent and reduces variancy, *i.e. a document without any colored tables may get patched with a colored one, leading to undesired learnings by the model.*
- We observed an absolute increase of `+0.0237` to the weighted average F1 score on table detection with `PVT v2 B4` as a backbone, and thus, decided to stick with clustering.

You can cluster data by executing `clusters.py`. Example usage:
```python
python clusters.py --f path/to/images/to/cluster \
                   --k number-of-clusters \   # must be an integer
                   --m    # move instead of copying
```

### Patching
Patching of tables to input images can be characterized by the following pipeline:
- Find the largest area on the input image that isn't already occupied by another table. Largest area ensures that we mask our patches over as much of text/figures as possible.
- Pad the table images to the width of the image.
- Paste the table at a proper scale on the center of the largest area.
- Generate output masks and PASCAL-VOC annotations.
You can generate augmented data by executing generator.py. Example usage:
```python
python generator.py --input-dir path/to/training/images \
                    --input-masks path/to/training/masks \
                    --patch-dir path/to/tables/to/patch \
                    --output-dir path/to/generated/data
```