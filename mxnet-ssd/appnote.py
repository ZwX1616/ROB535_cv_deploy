
### ipynbs ###
docker run -it --rm -p 4444:8888 -v /home/david/fashionAI/build-datasets-from-zero:/notebooks -v /data/david/open-image-v4:/mnt/openimgv4 registry.cn-shenzhen.aliyuncs.com/deeplearn/jupyter-py3:latest bash

pip install opencv-python
pip install mxnet
jupyter notebook --port=8888 --allow-root

#ssdtrain (new python3.5)
#docker run -it --rm -v /data/david/open-image-v4:/mnt/openimgv4 mxnet/python:1.2.0_gpu_cuda9 bash
#import pdb; pdb.set_trace()


## ---- mxnet-ssd docker ----

### training with valid set ###
docker run -it --rm -v /home/chuengwaising/ws/mxnet-ssd:/mnt -v /data/david/open-image-v4/_legacy_valid/val_dat_rec:/dat -v /data/weixing/pp_det_result:/res mxnet-cu90-ssd:v0.1 bash
# docker run -it --rm -v /home/chuengwaising/ws/mxnet-ssd:/mnt -v /data/david/open-image-v4/_legacy_valid/val_dat_rec:/dat -v /data/weixing/pp_det_result:/res mxnet/python:1.2.0_gpu_cuda9-dev
### training with main training set ###
# docker run -it --rm -v /home/chuengwaising/ws/mxnet-ssd:/mnt -v /data/david/open-image-v4/sources/rec_files:/dat -v /data/weixing/det_result:/res mxnet-cu90-ssd:v0.1 bash
docker run -it --rm -v /home/chuengwaising/ws/mxnet-ssd:/mnt -v /data/david/open-image-v4/sources/rec_files:/dat -v /data/weixing/det_result:/res mxnet/python:1.2.0_gpu_cuda9-dev bash
### (transfer) training pedestrian detector on 192.168.99.33 ###
docker run -it --rm -v /data/david/pedestrian_detection/mxnet-ssd-pedestrian:/mnt -v /data/david/pedestrian_detection/VOCdevkit/build:/dat -v /data/weixing:/res mxnet-cu90-ssd:v0.1 bash

#### result (person)(first 41 epochs on full openimage set) ####
#mobilenet with no OA
initial AP: 0.5947210235415363
transferred AP: 0.4498404818954341


## ---- train settings ----
```
person_labels=['/m/04yx4', '/m/03bt1vf', '/m/01bl7v', '/m/05r655', '/m/01g317'] #merger for 'person'
bag_labels=['/m/0hf58v5', '/m/01940j'] #merger for 'bag'
```

cd ../mnt


### training with un-grouped data in val (ng) ###
#### train ####
python3.6 train.py --train-path /dat/val_dat_pp_merge_ng.rec --val-path /dat/val_val_dat_pp_merge_ng.rec --prefix /res/densenet_tiny_ng/densenet_tiny --data-shape 512 512 --label-width 512 --lr 0.005 --network densenet-tiny --tensorboard True --num-class 1 --class-names 'Person' --gpu 5 --batch-size 16
python3.6 train.py --train-path /dat/val_dat_pp_merge_ng.rec --val-path /dat/val_val_dat_pp_merge_ng.rec --prefix /res/mobilenet_ng/mobilenet --data-shape 512 512 --label-width 512 --lr 0.01 --network mobilenet --tensorboard True --num-class 1 --class-names 'Person' --gpu 4 --batch-size 16

#### evaluate ####
python3.6 evaluate.py --rec-path /dat/val_val_dat_pp_merge_ng.rec --data-shape 512 512 --prefix /res/densenet_tiny/densenet_tiny_ng --network densenet-tiny --num-class 1 --class-names 'Person' --gpus 3 --batch-size 1 --epoch 239
python3.6 evaluate.py --rec-path /dat/val_val_dat_pp_merge_ng.rec --data-shape 512 512 --prefix /res/mobilenet_ng/mobilenet --network mobilenet --num-class 1 --class-names 'Person' --gpus 3 --batch-size 1 --epoch 239

### training with full data ###
#### train ####
python3.6 train.py --train-path /dat/train_person_and_bag.rec --val-path /dat/valid_person_and_bag.rec --prefix /res/mobilenet_pp+b/mobilenet --data-shape 512 384 --label-width 1440 --lr 0.005 --network mobilenet --tensorboard True --num-class 2 --class-names 'Person, Bag' --gpu 3 --batch-size 16 

AP (41): 0.3375491106153415

# densenet-tiny with OA
python3.6 train.py --train-path /dat/train_person_and_bag.rec --val-path /dat/valid_person_and_bag.rec --prefix /res/dt_person_oa/dt_person --data-shape 512 384 --label-width 1440 --lr 0.01 --network densenet-tiny-person --tensorboard True --num-class 2 --class-names 'Person, Bag' --gpu 1 --batch-size 12

initial AP(12): 0.49373230516953365




#### evaluate ####
python3.6 evaluate.py --rec-path /dat/valid_person_and_bag.rec --data-shape 512 384 --prefix /res/mobilenet_pp+b/mobilenet --network mobilenet --num-class 2 --class-names 'Person, Bag' --gpus 1 --batch-size 16 --epoch 41

#### demo ####
python3.6 demo.py --network mobilenet --epoch 41 --prefix /res/mobilenet_pp+b/mobilenet --data-shape 512 --class-names 'Person, Bag' --gpu 6 --thresh 0.5 --images './test_images/42.jpg'


## ---- tensorboard visualization (in shell) ----
cd (prefix fir)
tensorboard --ip 0.0.0.0 --logdir=logs


## ---- some demo ----
### densenet-tiny on val w/wo anchor optimization ###
#### train ####
python3.6 train.py --train-path /dat/val_dat_pp_merge_ng.rec --val-path /dat/val_val_dat_pp_merge_ng.rec --prefix /res/dt/dt --data-shape 512 512 --label-width 512 --lr 0.01 --network densenet-tiny --tensorboard True --num-class 1 --class-names 'Person' --gpu 4 --batch-size 16
python3.6 train.py --train-path /dat/val_dat_pp_merge_ng.rec --val-path /dat/val_val_dat_pp_merge_ng.rec --prefix /res/dt-oa/dt-oa --data-shape 512 512 --label-width 512 --lr 0.01 --network densenet-tiny-oa --tensorboard True --num-class 1 --class-names 'Person' --gpu 5 --batch-size 16

#### evaluate ####
python3.6 evaluate.py --rec-path /dat/val_val_dat_pp_merge_ng.rec --data-shape 512 512 --prefix /res/dt/dt --network densenet-tiny --num-class 1 --class-names 'Person' --gpus 4 --batch-size 16 --epoch 200
python3.6 evaluate.py --rec-path /dat/val_val_dat_pp_merge_ng.rec --data-shape 512 512 --prefix /res/dt-oa/dt-oa --network densenet-tiny-oa --num-class 1 --class-names 'Person' --gpus 4 --batch-size 16 --epoch 200

#### demo ####
python3.6 demo.py --network densenet-tiny --epoch 200 --prefix /res/dt/dt --data-shape 512 --class-names 'Person' --gpu 6 --thresh 0.5 --images './test_images/11.jpg'
python3.6 demo.py --network densenet-tiny-oa --epoch 200 --prefix /res/dt-oa/dt-oa --data-shape 512 --class-names 'Person' --gpu 6 --thresh 0.5 --images './test_images/12.jpg'

#### result ####
Before--mAP: 0.33132251165603543
After--mAP: 0.3501851190459783

## added 180718: CoordConv Test ##
# python3.6 train.py --train-path /dat/val_dat_pp_merge_ng.rec --val-path /dat/val_val_dat_pp_merge_ng.rec --prefix /res/dt-oa-cc/dt-oa-cc --data-shape 512 512 --label-width 512 --lr 0.01 --network densenet-tiny-person-cc --tensorboard True --num-class 1 --class-names 'Person' --gpu 1 --batch-size 12 --lr-factor 0.316 --lr-steps '5, 10, 15, 20, 25, 30, 35, 40 ,45' --end-epoch 50
python3.6 train.py --train-path /dat/val_dat_pp_merge_ng.rec --val-path /dat/val_val_d
at_pp_merge_ng.rec --prefix /res/dt-oa-cc/dt-oa-cc --data-shape 512 512 --label-width 512 --lr 0.01 --network
densenet-tiny-person-cc --tensorboard True --num-class 1 --class-names 'Person' --gpu 2,3,4,5,6,7 --batch-size
 60  --lr-factor 0.316 --lr-steps ' 100, 160 ,200 ,240' --end-epoch 280

python3.6 evaluate.py --rec-path /dat/val_val_dat_pp_merge_ng.rec --data-shape 512 512 --prefix /res/dt-oa-cc/dt-oa-cc --network densenet-tiny-person-cc --num-class 1 --class-names 'Person' --gpus 0 --batch-size 16 --epoch 200

### mobilenet on full bag training subset w/wo OA ###
#### train ####
python3.6 train.py --train-path /dat/train_bag.rec --val-path /dat/valid_bag.rec --prefix /res/mn_bag/mn_bag --data-shape 512 384 --label-width 1440 --lr 0.005 --network mobilenet --tensorboard True --num-class 1 --class-names 'Bag' --gpu 6 --batch-size 16
# python3.6 train.py --train-path /dat/train_bag.rec --val-path /dat/valid_bag.rec --prefix /res/mn_bag_oa/mn_bag_oa --data-shape 512 384 --label-width 1440 --lr 0.005 --network mobilenet_bag --tensorboard True --num-class 1 --class-names 'Bag' --gpu 7 --batch-size 16

python3.6 train.py --train-path /dat/train_bag.rec --val-path /dat/valid_bag.rec --prefix /res/dt_bag/dt_bag --data-shape 512 384 --label-width 1440 --lr 0.005 --network densenet-tiny --tensorboard True --num-class 1 --class-names 'Bag' --gpu 6 --batch-size 16
# python3.6 train.py --train-path /dat/train_bag.rec --val-path /dat/valid_bag.rec --prefix /res/dt_bag_oa/dt_bag_oa --data-shape 512 384 --label-width 1440 --lr 0.005 --network densenet-tiny-bag --tensorboard True --num-class 1 --class-names 'Bag' --gpu 7 --batch-size 16

#### evaluate ####
# python3.6 evaluate.py --rec-path /dat/valid_bag.rec --data-shape 512 384 --prefix /res/mn_bag/mn_bag --network mobilenet --num-class 1 --class-names 'Bag' --gpus 6 --batch-size 16 --epoch 239

python3.6 evaluate.py --rec-path /dat/valid_bag.rec --data-shape 512 384 --prefix /res/dt_bag/dt_bag --network densenet-tiny --num-class 1 --class-names 'Bag' --gpus 6 --batch-size 16 --epoch 239
# python3.6 evaluate.py --rec-path /dat/valid_bag.rec --data-shape 512 384 --prefix /res/dt_bag_oa/dt_bag_oa --network densenet-tiny-bag --num-class 1 --class-names 'Bag' --gpus 7 --batch-size 16 --epoch 239

#### demo ####
python3.6 demo.py --network densenet-tiny --epoch 239 --prefix /res/dt_bag/dt_bag --data-shape 512 --class-names 'Bag' --gpu 6 --thresh 0.5 --images './test_images/001.jpg'
# python3.6 demo.py --network densenet-tiny-oa --epoch 200 --prefix /res/dt-oa/dt-oa --data-shape 512 --class-names 'Person' --gpu 6 --thresh 0.5 --images './test_images/12.jpg'

#### result ####
Before--mAP: 0.35968379446640314 (backpack only: 0.2727272727272727)
# After--mAP: 0.3501851190459783



