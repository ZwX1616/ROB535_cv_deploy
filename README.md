# ROB535_cv_deploy
_for in-class competition task1 &amp; task2_

curated by wx. zhang

Results:
![](https://github.com/ZwX1616/ROB535_cv_deploy/blob/master/acc.jpg)

Environment requirements:

0. Python 3.6
1. packages: mxnet, gluonbook, gluoncv, pandas, ...
2. preferably mxnet-cu92 to use CUDA accerleration (mxnet-cu90 is recommended if you use tf)
3. if not using GPU, change all ctx to mx.cpu()

Directory strcuture:
```
root
├── data
│   ├── trainval
│       ├── 0cec3d1f-544c-4146-8632-84b1f9fe89d3
│           ├── (image, bbox, cloud, proj files)
│       ├── ...
│   ├── test
│       ├── 0ff0a23e-5f50-4461-8ccf-2b71bead2e8e
│           ├── (image, cloud, proj files)
│       ├── ...
│   ├── train.rec (7000 512*384 images + bbox labels)
│   ├── train.lst (bbox labels and filenames)
│   ├── valid.rec (573 512*384 images + bbox labels)
│   ├── valid.lst (bbox labels and filenames)
│   ├── train_v3.rec (7000 768*432 images + class labels)
│   ├── train_v3.lst (class labels and filenames)
│   ├── valid_v3.rec (573 768*432 images + class labels)
│   ├── valid_v3.lst (class labels and filenames)
│   ├── test_v3.rec (2631 768*432 images)
│   ├── test_v3.lst (a list of filenames)
│   ├── xyz_train.txt (XYZ coordinates training data)
│   ├── xyz_valid.txt (XYZ coordinates training data)
│   │
│   ├── classes.csv (class 0,1,2)
│   ├── classes_v2.csv (class 0,1,2,3,4,5)
├── mxnet-dbc
    ├── ...
├── mxnet-mlp
    ├── ...
├── mxnet-ssd (originally forked from https://github.com/zhreshold/mxnet-ssd)
    ├── ...
├── params
    ├── (network parameters for ssd and frcnn)
├── output
    ├── (txt files for kaggle upload)
├── README.md
```
(Note:.rec file is the mxnet RecordIO format, refer to https://github.com/leocvml/mxnet-im2rec_tutorial on how to generate them)

for task1:
- train SSD: python mxnet-ssd/gta5_train.py
- predict with SSD: python mxnet-ssd/gta5_predict.py 42
- train Inception3 classifier: python mxnet-dbc/train.py
- predict with Inception3 classifier: python mxnet-dbc/predict.py

for task2:
- train MLP: python mxnet-mlp/mlp_train.py
- (after predicting with SSD) predict with MLP: python mxnet-mlp/predict.py
