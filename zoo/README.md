# Segmentation Models Zoo

Collection of semantic segmentation models trained on different [datasets](#datasets).

## How to use

Models zoo organized using [DVC](https://dvc.org) and DVCX tools for managment of models code together with models weights. It allows to load model even if repo have been updated and backward compatability of weights have been broken, all you need is to specify which revision to use (listed bellow in tables for each model).  

Here is a code snippet how to load model with `dvcx` (select `name`, `revision` and `dvcsummon` file in one of the tables below):
 ```python
import dvcx

model = dvcx.summon(
    <name>, 
    rev=<revison>, 
    summon_file=<summon_file>,
    repo="https://github.com/qubvel/segmentation_models.pytorch",
)
```
[Notebook](../examples/pretrained%20models%20inference.ipynb) with example of model loading and inference. 

## Datasets

 - [ADE20K](http://sceneparsing.csail.mit.edu/) [[models](#ade20k)]
 - [COCO-Stuff](http://cocodataset.org/#stuff-eval) [[models](#coco-stuff)]
 - [CamVid](https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) [[models](#camvid)]
 - [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) [[models](#pascal-voc)]
 - [CityScapes](https://www.cityscapes-dataset.com/) [[models](#cityscapes)]
 - [Inria Aerial Image Labeling](https://project.inria.fr/aerialimagelabeling/) [[models](#inria)]
 
 ---
 
 ### ADE20K
 **Dataset description:**  
 There are totally 150 semantic categories, which include stuffs like sky, road, grass, and discrete objects like person, car, bed. Note that there are non-uniform distribution of objects occuring in the images, mimicking a more natural object occurrence in daily scene.

Labels info is available [here](https://github.com/CSAILVision/placeschallenge/blob/master/sceneparsing/objectInfo150.txt).

<details>
<summary>Code snippet for model loading.</summary>
<p>

```python
name =   # paste model name here
rev =    # paste revison here
summon_file =   # paste summon file name here
repo = "https://github.com/qubvel/segmentation_models.pytorch/"

model = dvcx.summon(name, rev=rev, repo=repo, summon_file=summon_file)
```
</p>
</details>

<details>
<summary>Image preprocessing (if another is not specified in table).</summary>
<p>

```python
mean = np.array([123.67, 116.28, 103.53])
std = np.array([58.39, 57.12, 57.37])
preprocessed_image = (image - mean) / std
```
</p>
</details>

| Name                                | mIoU score | Pixel Acc. | Revision | Summon File     | Image size      | FP16** |
|-------------------------------------|:----------:|:----------:|:--------:|:---------------:|:---------------:|:------:|
| 001_ade20k\_fpn\_srx50              | 38.24      | 77.33      | master   | zoo/ade20k.yaml | 512\*           | -      |
| 011_ade20k_fpn_se_resnext101_32x4d  | 40.93      | 77.82      | master   | zoo/ade20k.yaml | 512\*           | -      |
| 019_ade20k_fpn_se_resnext50_32x4d   | 39.97      | 77.12      | master   | zoo/ade20k.yaml | 512\*           | -      |
| 020_ade20k_fpn_senet154             | 43.30      | 78.13      | master   | zoo/ade20k.yaml | 512\*           | O1     |
| 023_ade20k_fpn_efficientnet-b7      | 46.73      | 80.25      | master   | zoo/ade20k.yaml | 512\*           | O1     |
| 024_ade20k_fpn_efficientnet-b5      | 44.46      | 79.28      | master   | zoo/ade20k.yaml | 512\*           | O1     |

\* - size of smallest image dimension (preserving aspect ratio)
\** - opt_level according to NVIDIA Apex tool 

scores reported for validation set **without** test time augmentation and multiscale testing

---

 ### COCO-Stuff
 **Dataset description:**  
 The COCO Stuff Segmentation Task is designed to push the state of the art in semantic segmentation of stuff classes. Whereas the object detection task addresses thing classes (person, car, elephant), this task focuses on stuff classes (grass, wall, sky).

Labels info is available [here](https://github.com/nightrome/cocostuff/blob/master/labels.md).

<details>
<summary>Code snippet for model loading.</summary>
<p>

```python
name =   # paste model name here
rev =    # paste revison here
summon_file =   # paste summon file name here
repo = "https://github.com/qubvel/segmentation_models.pytorch/"

model = dvcx.summon(name, rev=rev, repo=repo, summon_file=summon_file)
```
</p>
</details>

<details>
<summary>Image preprocessing (if another is not specified in table).</summary>
<p>

```python
mean = np.array([123.67, 116.28, 103.53])
std = np.array([58.39, 57.12, 57.37])
preprocessed_image = (image - mean) / std
```
</p>
</details>

| Name                        | mIoU score | Pixel Acc\. | Revision | Summon File     |Train crop size  |
|-----------------------------|:----------:|:-----------:|:--------:|:---------------:|:---------------:|
| 001_coco-stuff\_fpn\_srx50  | 41.94      | 64.73       | master   | zoo/coco.yaml   | 512x512         |

scores reported for validation set **without** test time augmentation and multiscale testing

---

 ### CamVid
 **Dataset description:**  
 The Cambridge-driving Labeled Video Database (CamVid) is the first collection of videos with object class semantic labels, complete with metadata.

Labels map ```['unlabelled', 'sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist']```

<details>
<summary>Code snippet for model loading.</summary>
<p>

```python
name =   # paste model name here
rev =    # paste revison here
summon_file =   # paste summon file name here
repo = "https://github.com/qubvel/segmentation_models.pytorch/"

model = dvcx.summon(name, rev=rev, repo=repo, summon_file=summon_file)
```
</p>
</details>

<details>
<summary>Image preprocessing (if another is not specified in table).</summary>
<p>

```python
mean = np.array([123.67, 116.28, 103.53])
std = np.array([58.39, 57.12, 57.37])
preprocessed_image = (image - mean) / std
```
</p>
</details>

| Name                    | mIoU score | Pixel Acc\. | Revision | Summon File     | Image size |
|-------------------------|:----------:|:-----------:|:--------:|:---------------:|:----------:|
| 019_camvid_unet_se_resnext50_32x4d   | 75.13 | 94.47 | master | zoo/camvid.yaml | 360 x 480 |
| 020_camvid_unet_se_resnext50_32x4d   | 77.53 | 95.07 | master | zoo/camvid.yaml | 512 x 672 |
| 021_camvid_unet_se_resnext101_32x4d  | 78.03 | 95.68 | master | zoo/camvid.yaml | 512 x 672 |
| 022_camvid_unet_efficientnet-b3      | 75.08 | 95.08 | master | zoo/camvid.yaml | 512 x 672 |
| 022_camvid_fpn_efficientnet-b5       | 77.39 | 95.38 | master | zoo/camvid.yaml | 512 x 672 |
| 024_camvid_pan_resnext50_32x4d       | 75.42 | 94,36 | master | zoo/camvid.yaml | 512 x 672 |
| 025_camvid_deeplabv3_resnext50_32x4d | 78.11 | 95.30 | master | zoo/camvid.yaml | 512 x 672 |
| 027_camvid_pspnet_se_resnext50_32x4d | 73.02 | 94.59 | master | zoo/camvid.yaml | 512 x 672 |


scores reported for validation set **without** test time augmentation and multiscale testing


---

 ### Pascal VOC
 **Dataset description:**  
The main goal of this challenge is to recognize objects from a number of visual object classes in realistic scenes (i.e. not pre-segmented objects). It is fundamentally a supervised learning learning problem in that a training set of labelled images is provided. The twenty object classes that have been selected are:

 - Person: person
 - Animal: bird, cat, cow, dog, horse, sheep
 - Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
 - Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor

Labels map ```['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']```

<details>
<summary>Code snippet for model loading.</summary>
<p>

```python
name =   # paste model name here
rev =    # paste revison here
summon_file =   # paste summon file name here
repo = "https://github.com/qubvel/segmentation_models.pytorch/"

model = dvcx.summon(name, rev=rev, repo=repo, summon_file=summon_file)
```
</p>
</details>

<details>
<summary>Image preprocessing (if another is not specified in table).</summary>
<p>

```python
mean = np.array([123.67, 116.28, 103.53])
std = np.array([58.39, 57.12, 57.37])
preprocessed_image = (image - mean) / std
```
</p>
</details>

| Name                                    | mIoU score | Pixel Acc\. | Revision | Summon File     | Image size | FP16** |
|-----------------------------------------|:----------:|:-----------:|:--------:|:---------------:|:----------:|:------:|
| 001_voc_fpn_resnet34                    | 64.11 | 91.39 | master | zoo/voc.yaml | 512* | O1 |
| 002_voc_deeplabv3_resnet34              | 67.65 | 92.25 | master | zoo/voc.yaml | 512* | O1 |
| 003_voc_fpn_efficientnet-b5             | 77.55 | 94.87 | master | zoo/voc.yaml | 512* | O1 |
| 005_voc_pan_efficientnet-b5             | 72.74 | 93.62 | master | zoo/voc.yaml | 512* | O1 |
| 006_voc_deeplabv3plus_efficientnet-b5   | 77.55 | 94.76 | master | zoo/voc.yaml | 512* | O1 |
| 007_voc_deeplabv3plus_efficientnet-b7   | 78.49 | 95.07 | master | zoo/voc.yaml | 512* | O1 |
| 013_voc_fpn_se_resnext50_32x4d          | 71.51 | 93.45 | master | zoo/voc.yaml | 512* | O1 |
| 014_voc_fpn_efficientnet-b7             | 81.09 | 95.81 | master | zoo/voc.yaml | 512* | O1 |
| 015_voc_deeplabv3plus_efficientnet-b7   | 80.28 | 95.62 | master | zoo/voc.yaml | 512* | O1 |
| 016_voc_unet_efficientnet-b5            | 81.52 | 95.89 | master | zoo/voc.yaml | 512* | O1 |
| 017_voc_unet_efficientnet-b7            | 81.47 | 95.91 | master | zoo/voc.yaml | 512* | O1 |
| 018_voc_unet_se_resnext50_32x4d         | 70.09 | 93.10 | master | zoo/voc.yaml | 512* | O1 |
| 019_voc_unet_se_resnext101_32x4d        | 71.99 | 93.80 | master | zoo/voc.yaml | 512* | O1 |
| 021_voc_fpn_timm-efficientnet-b7        | 76.45 | 94.64 | master | zoo/voc.yaml | 512* | O1 |

\* - size of smallest image dimension (preserving aspect ratio)  
\** - opt_level according to NVIDIA Apex tool 

scores reported for validation set **without** test time augmentation and multiscale testing


