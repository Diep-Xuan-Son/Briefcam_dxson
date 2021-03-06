import math
import numpy as np
import torch
import torch.nn as nn
from models.common import Conv
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