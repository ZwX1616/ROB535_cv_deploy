from pathlib import Path
from glob import glob
import mxnet as mx
import numpy as np
from timeit import default_timer as timer
from dataset.testdb import TestDB
from dataset.iterator import DetIter
import csv
import os

class Detector(object):
	"""
	SSD detector which hold a detection network and wraps detection API

	Parameters:
	----------
	symbol : mx.Symbol
		detection network Symbol
	model_prefix : str
		name prefix of trained model
	epoch : int
		load epoch of trained model
	data_shape : int
		input data resize shape
	mean_pixels : tuple of float
		(mean_r, mean_g, mean_b)
	batch_size : int
		run detection with batch size
	ctx : mx.ctx
		device to use, if None, use mx.cpu() as default context
	"""
	def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels, \
				 batch_size=1, ctx=None):
		self.ctx = ctx
		if self.ctx is None:
			self.ctx = mx.cpu()
		load_symbol, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
		if symbol is None:
			symbol = load_symbol
		self.mod = mx.mod.Module(symbol, label_names=None, context=ctx)
		self.data_shape = data_shape
		self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape, data_shape))])
		self.mod.set_params(args, auxs)
		self.data_shape = data_shape
		self.mean_pixels = mean_pixels

	def detect(self, det_iter, show_timer=False):
		"""
		detect all images in iterator

		Parameters:
		----------
		det_iter : DetIter
			iterator for all testing images
		show_timer : Boolean
			whether to print out detection exec time

		Returns:
		----------
		list of detection results
		"""
		num_images = det_iter._size
		result = []
		detections = []
		if not isinstance(det_iter, mx.io.PrefetchingIter):
			det_iter = mx.io.PrefetchingIter(det_iter)
		start = timer()
		for pred, _, _ in self.mod.iter_predict(det_iter):
			detections.append(pred[0].asnumpy())
		time_elapsed = timer() - start
		if show_timer:
			print("Detection time for {} images: {:.4f} sec".format(
				num_images, time_elapsed))
		for output in detections:
			for i in range(output.shape[0]):
				det = output[i, :, :]
				res = det[np.where(det[:, 0] >= 0)[0]]
				result.append(res)
		return result

	def im_detect(self, im_list, root_dir=None, extension=None, show_timer=False):
		"""
		wrapper for detecting multiple images

		Parameters:
		----------
		im_list : list of str
			image path or list of image paths
		root_dir : str
			directory of input images, optional if image path already
			has full directory information
		extension : str
			image extension, eg. ".jpg", optional

		Returns:
		----------
		list of detection results in format [det0, det1...], det is in
		format np.array([id, score, xmin, ymin, xmax, ymax]...)
		"""
		test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
		test_iter = DetIter(test_db, 1, self.data_shape, self.mean_pixels,
							is_train=False)
		return self.detect(test_iter, show_timer)

	def detect_and_save_result (self, im_list, root_dir, extension=None,
							 classes=[], thresh=0.6, show_timer=False):
		
		import cv2
		dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
		if not isinstance(im_list, list):
			im_list = [im_list]
		assert len(dets) == len(im_list)

		filelist=glob('../data/test/*/*_image.jpg')
		wf=open('../output_t1.txt','w+',newline='')
		writer=csv.writer(wf)
		writer.writerow(['guid/image','label']) ## for task1

		cls_dict={0:1,1:2}
		nullcount=0

		for k, det in enumerate(dets):
			print(str(k),end='\r')
			filename=str(filelist[k])
			img=filename[-10-4:-10]
			guid=filename[-10-4-1-36:-10-4-1]

			if len(det)==0: 
				lbl=str(1)
				nullcount+=1
			else:
				print('dets: '+str(det.shape[0]),end='\r')
				conf=np.zeros(det.shape[0])
				for j in range(det.shape[0]):
					conf[j]=float(det[j][1])
				best_idx=np.argmax(conf)
				lbl=str(cls_dict[int(det[best_idx][0])])
			writer.writerow([guid+'/'+img,lbl]) ## for task1
			
			last_idx=k
		print('udgl='+str(nullcount)+'/'+str(last_idx+1))