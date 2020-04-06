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

| Name                    | mIoU score | Pixel Acc\. | Revision | Summon File     | Train crop size |
|-------------------------|:----------:|:-----------:|:--------:|:---------------:|:---------------:|
| 001_ade20k\_fpn\_srx50  | 38\.24     | 77\.33      | master   | zoo/ade20k.yaml | 512x512         |

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
|-------------------------|:----------:|:-----------:|:--------:|:---------------:|:---------------:|
| 021_camvid_unet_se_resnext101_32x4d| 78.03 | 95.68 | master | zoo/camvid.yaml | 512 x 672 |
| 020_camvid_unet_se_resnext50_32x4d | 77.53 | 95.07 | master | zoo/camvid.yaml | 512 x 672 |
| 019_camvid_unet_se_resnext50_32x4d | 75.13 | 94.47 | master | zoo/camvid.yaml | 360 x 480 |


scores reported for validation set **without** test time augmentation and multiscale testing


