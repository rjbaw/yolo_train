# YOLOv3 Training Script
## Datasets
`https://www.dropbox.com/sh/u5c86zbemh7mz3v/AABPm4dVrzIse1WYf71WQIUQa?dl=0`
## Setup
`git clone https://github.com/ezvk7740/yolo_train.git`
### Dataset Configuration
Create empty configuration files
```
touch data/${DATASET_NAME}.data
touch data/${DATASET_NAME}.names
```
#### ${DATASET_NAME}.data
```
classes=${NUMBER_OF CLASSES}
train=data/${DATASET_NAME}/train
valid=data/${DATASET_NAME}/val
names=data/${DATASET_NAME}.names
backup=backup/
eval=${DATASET_NAME}
```
#### ${DATASET_NAME}.names
```
${CLASS_NAMES}
.
.
.
```
#### Dataset arrangement
Rename dataset files  
`python data/rename.py --data ${DATASET_NAME}`
Proportion dataset files  
`python data/proportion.py --data ${DATASET_NAME} --proportion ${TRAIN_PROPORTION}`
## Running
`python train.py`

