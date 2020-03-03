# Segmentation Models Zoo

To get the model using `dvcx` select the appropriate `name`, `revision` and `dvcsummon` file in the tables below:
 ```python
import dvcx

model = dvcx.summon(
    <name>, 
    rev=<revison>, 
    summon_file=<summon_file>,
    repo="https://github.com/qubvel/segmentation_models.pytorch",
)
```


### Datasets

 - [CamVid] [[models](#camvid)]
 - [COCO-Stuff] [[models](#coco-stuff)]
 - [ADE20K] [[models](#ade20k)]
 - [Pascal VOC] [[models](#pascal-voc)]
 - [CityScapes] [[models](#cityscapes)]
 - [Inria semantic labeling] [[models](#inria)]
 
 ### ADE20K
 **Dataset description:**  
 There are totally 150 semantic categories, which include stuffs like sky, road, grass, and discrete objects like person, car, bed. Note that there are non-uniform distribution of objects occuring in the images, mimicking a more natural object occurrence in daily scene.

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

Labels info is available [here](https://github.com/CSAILVision/placeschallenge/blob/master/sceneparsing/objectInfo150.txt).

<details>
<summary>Image preprocessing is standard (if another is not specified in table).</summary>
<p>

```python
mean = np.array([123.67, 116.28, 103.53])
std = np.array([58.39, 57.12, 57.37])
preprocessed_image = (image - mean) / std
```
</p>
</details>

| Name                    | mIoU score | Pixel Acc\. | Revision | Summon File     |
|-------------------------|:----------:|:-----------:|:--------:|:---------------:|
| 001_ade20k\_fpn\_srx50  | 38\.24     | 77\.33      | master   | zoo/ade20k.yaml |


 ### COCO-Stuff
 **Dataset description:**  
 The COCO Stuff Segmentation Task is designed to push the state of the art in semantic segmentation of stuff classes. Whereas the object detection task addresses thing classes (person, car, elephant), this task focuses on stuff classes (grass, wall, sky).

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

Labels info is available [here](https://github.com/nightrome/cocostuff/blob/master/labels.md).

<details>
<summary>Image preprocessing is standard (if another is not specified in table).</summary>
<p>

```python
mean = np.array([123.67, 116.28, 103.53])
std = np.array([58.39, 57.12, 57.37])
preprocessed_image = (image - mean) / std
```
</p>
</details>

| Name                    | mIoU score | Pixel Acc\. | Revision | Summon File     |
|-------------------------|:----------:|:-----------:|:--------:|:---------------:|
| 001_coco-stuff\_fpn\_srx50  |      |       | master   | zoo/coco.yaml |


 ### CamVid
 **Dataset description:**  
 The Cambridge-driving Labeled Video Database (CamVid) is the first collection of videos with object class semantic labels, complete with metadata.

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

Labels map ```['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled']```

<details>
<summary>Image preprocessing is standard (if another is not specified in table).</summary>
<p>

```python
mean = np.array([123.67, 116.28, 103.53])
std = np.array([58.39, 57.12, 57.37])
preprocessed_image = (image - mean) / std
```
</p>
</details>

| Name                    | mIoU score | Pixel Acc\. | Revision | Summon File     |
|-------------------------|:----------:|:-----------:|:--------:|:---------------:|
| 001_camvid\_fpn\_srx50  |      |       | master   | zoo/camvid.yaml |

