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
 
| Name                    | mIoU score | Pixel Acc\. | Revision | Summon File     |
|-------------------------|:----------:|:-----------:|:--------:|:---------------:|
| 001_ade20k\_fpn\_srx50  | 38\.24     | 77\.33      | master   | zoo/ade20k.yaml |
|                         |            |             |          ||
|                         |            |             |          ||
|                         |            |             |          ||

 