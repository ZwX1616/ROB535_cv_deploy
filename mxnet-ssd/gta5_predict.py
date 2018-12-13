
# predict and save results in the submission format

import os
import sys
import csv
from pathlib import Path
from glob import glob
import argparse
import mxnet as mx

import tools.find_mxnet
from detect.detector import Detector
from symbol.symbol_factory import get_symbol


test_all=1 # should be True 


# prediction parameters #

thresh=0.2 # you may want to change this

network='densenet-tiny-oi-cc'
nms_thresh=0.5
force_nms=True
data_shape=512
background_id=-1
show_timer=True
# ------------------ #

def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx, num_class,
                 nms_thresh=0.5, force_nms=True, nms_topk=400,
                 threshold=0.6, background_id=-1):
    """
    wrapper for initialize a detector

    Parameters:
    ----------
    net : str
        test network name
    prefix : str
        load model prefix
    epoch : int
        load model epoch
    data_shape : int
        resize image shape
    mean_pixels : tuple (float, float, float)
        mean pixel values (R, G, B)
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(?)
    num_class : int
        number of classes
    nms_thresh : float
        non-maximum suppression threshold
    force_nms : bool
        force suppress different categories
    """
    if net is not None:
        net = get_symbol(net, data_shape, num_classes=num_class, nms_thresh=nms_thresh,
            force_nms=force_nms, nms_topk=nms_topk,
            threshold=threshold, background_id=background_id)
    detector = Detector(net, prefix, epoch, data_shape, mean_pixels, ctx=ctx)
    return detector


if not int(len(sys.argv))==2:

    print('ONE arguments are required: ')
    print('-arg1 needs to be the epoch of your trained model')
    sys.exit()

else:
    ctx = mx.cpu() #

    prefix='../params/gta5_2d'
    epoch=int(sys.argv[1])

    class_names=['0','1','2']
    image_list=[str(i.as_posix()) for i in Path('../data/test/').glob("*/*.jpg")]
    #import pdb; pdb.set_trace() ###
    print('found '+str(len(image_list))+' test images...')
    if not test_all:
        image_list=image_list[0:10]
    print('testing '+str(len(image_list))+' test images...')

    detector = get_detector(network, prefix, epoch,
                            data_shape,
                            (123, 117, 104),
                            ctx, len(class_names), nms_thresh, force_nms, nms_topk=20,
                            threshold=thresh, background_id=background_id)
    # run detection
    detector.detect_and_save_result(image_list, '.', None,
                                  class_names, thresh, show_timer)
