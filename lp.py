import argparse
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier
from utils.datasets import letterbox
import os
from tqdm import tqdm
from PIL import Image
import random as rd
def detect_obj(model, stride, names, img_detect = '', draw = False, iou_thres = 0.4, conf_thres = 0.5, img_size = 640):
	global num_img
	global color
	imgsz = img_size
	high, weight = img_detect.shape[:2]
	check = False
	#####################################
	classify = False
	agnostic_nms = False
	augment = False
	# Set Dataloader
	#vid_path, vid_writer = None, None
	# Get names and colors
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	half = torch.cuda.is_available()
	count = 0
	t = time.time()
	#processing images
	'''
	Tiền xử lí ảnh
	'''
	im0 = letterbox(img_detect, img_size, stride= 32)[0]
	im0 = im0[:, :, ::-1].transpose(2, 0, 1)
	im0 = np.ascontiguousarray(im0)
	im0 = torch.from_numpy(im0).to(device)
	im0 = im0.half() if half else im0.float()
	im0 /= 255.0  # 0 - 255 to 0.0 - 1.0
	if im0.ndimension() == 3:
		im0 = im0.unsqueeze(0)
	# Inference
	pred = model(im0, augment= augment)[0]
	# Apply NMS
	classes = 0 # phuong fix classes = None
	pred = non_max_suppression(pred, conf_thres, iou_thres, classes = classes, agnostic=agnostic_nms)
	# Apply Classifier
	if classify:
		pred = apply_classifier(pred, modelc, im0, img_ocr)
	gn = torch.tensor(img_detect.shape)[[1, 0, 1, 0]]# normalization gain whwh
	points = []
	if len(pred[0]):
		check = True
		pred[0][:, :4] = scale_coords(im0.shape[2:], pred[0][:, :4], img_detect.shape).round()
		for c in pred[0][:, -1].unique():
			n = (pred[0][:, -1] == c).sum()  # detections per class
		# print(pred[0])
		for box in pred[0]:
			c1 = (int(box[0]), int(box[1]))
			c2 = (int(box[2]), int(box[3]))
			x1, y1 = c1
			x2, y2 = c2
			acc = round(float(box[4]),2)
			cls = int(box[5])
			label = names[cls]#
			img_crop = img_detect[y1:y2, x1:x2]
			if draw:
				cv2.rectangle(img_detect, c1, c2, (255, 0, 0), 2)
			points.append([x1, y1, x2, y2, acc, cls])
	return img_detect, points