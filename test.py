import math
import numpy as np
import torch
import torch.nn as nn
from models.common import Conv
import os
from net import get_model
import json
from torchvision import transforms as T
import cv2
import time
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier
from utils.datasets import letterbox
from PIL import Image
class PROCESS():
	def __init__(self, model_obj, stride, names, model_ft):
		self.model_obj = model_obj
		self.model_ft = model_ft
		self.stride = stride
		self.names = names
		self.transforms = T.Compose([T.Resize(size=(288, 144)),T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
		self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		self.half = torch.cuda.is_available()
		self.gallery = dict()
	def predict_att(self, img):
		src = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		src = self.transforms(src)
		src = src.unsqueeze(dim=0)
		out = self.model_ft.forward(src)
		pred = torch.gt(out, torch.ones_like(out)*0.5).int()  # threshold=0.5
		return pred
	def detect_obj(self, img_detect = '', iou_thres = 0.4, conf_thres = 0.5, img_size = 640):
		high, weight = img_detect.shape[:2]
		check = False
		classify = False
		agnostic_nms = False
		augment = False
		count = 0
		t = time.time()
		im0 = letterbox(img_detect, img_size, stride= self.stride)[0]
		im0 = im0[:, :, ::-1].transpose(2, 0, 1)
		im0 = np.ascontiguousarray(im0)
		im0 = torch.from_numpy(im0).to(self.device)
		im0 = im0.half() if half else im0.float()
		im0 /= 255.0  # 0 - 255 to 0.0 - 1.0
		if im0.ndimension() == 3:
			im0 = im0.unsqueeze(0)
		# Inference
		pred = self.model_obj(im0, augment= augment)[0]
		# Apply NMS
		classes = None
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
			for box in pred[0]:
				c1 = (int(box[0]), int(box[1]))
				c2 = (int(box[2]), int(box[3]))
				x1, y1 = c1
				x2, y2 = c2
				acc = round(float(box[4])*100,2)
				cls = int(box[5])
				label = self.names[cls]#
				img_crop = img_detect[y1:y2, x1:x2]
				if label == 'person':
					result  = self.predict_att(img_crop)
					print(result)
					
		return img_detect, points
class Ensemble(nn.ModuleList):
	# Ensemble of models
	def __init__(self):
		super().__init__()
	def forward(self, x, augment=False, profile=False, visualize=False):
		y = []
		for module in self:
			y.append(module(x, augment, profile, visualize)[0])
		y = torch.cat(y, 1)
		return y, None
def load_model(weights, map_location=None, inplace=True, fuse=True):
	from models.yolo import Detect, Model
	# Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
	model = Ensemble()
	for w in weights if isinstance(weights, list) else [weights]:
		if not os.path.isfile(w):
			print('No have model file!')
			continue
		ckpt = torch.load(w, map_location=map_location)  # load
		if fuse:
			model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
		else:
			model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # without layer fuse
	# Compatibility updates
	for m in model.modules():
		if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
			m.inplace = inplace  # pytorch 1.7.0 compatibility
			if type(m) is Detect:
				if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
					delattr(m, 'anchor_grid')
					setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
		elif type(m) is Conv:
			m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
	if len(model) == 1:
		return model[-1]  # return model
	else:
		print(f'Ensemble created with {weights}\n')
		for k in ['names']:
			setattr(model, k, getattr(model[-1], k))
		model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
		return model  # return ensemble
def load_network(network):
	save_path = os.path.join('./weights', 'person.pth')
	network.load_state_dict(torch.load(save_path))
	print('Resume model from {}'.format(save_path))
	return network
if __name__ == '__main__':
	weights = './weights/yolov5s.pt'
	model = load_model(weights, map_location = 'cpu')
	model.eval()
	stride = int(model.stride.max())  # model stride
	names = model.module.names if hasattr(model, 'module') else model.names
	half = torch.cuda.is_available()
	##############################################################
	with open('./doc/final_label.json', 'r') as f:
		label  = list(json.load(f).values())
		num_label = len(label)
	label.sort()
	num_id = 1
	transforms = T.Compose([T.Resize(size=(288, 144)),T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	backbone = 'resnet50'
	model_name = '{}_nfc'.format(backbone)
	model_ft = get_model(model_name, num_label, num_id=num_id)
	model_ft = load_network(model_ft)
	model_ft.eval()
	if half:
		model.half()
	processing = PROCESS(model, stride, names, model_ft)
	cap = cv2.VideoCapture('test.webm')
	if not cap.isOpened():
		print('No camera input!')
	while True:
		ret, frame = cap.read()
		if ret:
			frame, points = processing.detect_obj(img_detect = frame)