import os
import numpy as np
import torch
import glob
import time
import cv2
import argparse
import sys
from utils.sort.sort import iou_batch
from lp import detect_obj
from models.experimental import attempt_load
import random as rd
import json
from torchvision import transforms as T
from net import get_model
from PIL import Image
from utils.progress import *
from deep_sort_pytorch.utils.parser import get_config
# from deep_sort_pytorch.deep_sort import DeepSort
from utils.general import xyxy2xywh
from utils.downloads import attempt_download
from Sort.sort import *
import sys
def int_0(a, w, h):
  x1, y1, x2, y2 = a
  if x1 < 0:
    x1 = 0
  if y1 < 0:
    y1 = 0
  if x2 > w:
    x2 = w
  if y2 > h:
    y2 = h
  return x1, y1, x2, y2
def load_network(network):
  save_path = os.path.join('./weights', 'person.pth')
  network.load_state_dict(torch.load(save_path))
  print('Resume model from {}'.format(save_path))
  return network
def compute_date(time_):
  sec = str(round(time_%60,2)).rjust(2, '0')
  minute = str(int(time_//60)).rjust(2, '0')
  hour = str(int(int(time_//60)//60)).rjust(2, '0')
  return f'{hour}:{minute}:{sec}'
if __name__ == '__main__':
  result = dict()
  data_img = dict()
  att_all = dict()
  time_result = dict()
  #Model nhận diện thuộc tính
  with open('./doc/person_final_label.json', 'r') as labels:
      label = json.load(labels)
      label= list(label.values())
  label.sort()
  num_label = len(label)
  num_id = 250
  num_id = 1
  transforms = T.Compose([
      T.Resize(size=(288, 144)),
      T.ToTensor(),
      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  # all train
  t1_loadmodel = time.time()
  args = parse_args()
  mot_tracker = Sort()
  # mot_tracker = Sort(max_age=args.max_age, 
  #                     min_hits=args.min_hits,
  #                     iou_threshold=args.iou_threshold) #create instance of the SORT tracke
  model_name = '{}_nfc_id'.format(args.backbone) if args.use_id else '{}_nfc'.format(args.backbone)
  model_att = get_model(model_name, num_label, use_id=args.use_id, num_id=num_id)
  model_att = load_network(model_att)
  model_att.eval()
  if torch.cuda.is_available():
    model_att.cuda()
  # Load model yolov5
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  half = torch.cuda.is_available()	# half precision only supported on CUDA
  # Load model nhan dien container
  # weights = '/content/drive/MyDrive/yolov5/runs/train/exp8/weights/best.pt'
  model = attempt_load(args.weights, map_location=device)	# load FP32 model
  stride = int(model.stride.max())	# model stride
  names = model.module.names if hasattr(model, 'module') else model.names
  if half:
    model.half()
  # khởi tạo deepsort
  # cfg = get_config()
  # cfg.merge_from_file(args.config_deepsort)
  # attempt_download(args.deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
  # deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
  #                     max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
  #                     max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
  #                     max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
  #                     use_cuda=True)
  print('Loaded model in %0.2f s'%(time.time()-t1_loadmodel))
  if args.input is None:
    print("No input video! Please set the video input!")
    sys.exit()
  cap = cv2.VideoCapture(args.input)
  print(args.input)
  if not cap.isOpened():
    print('Video is not available!')
  count = 0
  basename = os.path.basename(args.input).split('.')[0]
  dir_save = f'/content/result/{basename}'
  path_bgs = f"/content/result/{basename}_bgs"
  if not os.path.isdir(dir_save):
    os.mkdir(dir_save)
  if not os.path.isdir(path_bgs):
    os.mkdir(path_bgs)
  backgroundModels = []
  time_ = 0
  fps = 30
  t1 = time.time()
  images = []
  bgs_200 = []
  while True:
    ret, frame = cap.read()
    if ret:
      count += 1
      if count < 30000:
        continue
      if count % 40000 == 0 and count != 0:
        sys.exit(0)
      time_ = count/fps # Tính thời gian
      if count % 100 == 0:
        print(f'Processing in {count} s, FPS: {count/(time.time()-t1)}')
        t1 = time.time()
      if count % 7000 == 0:
        result['att'] = processing(att_all, label)
        result['data'] = data_img
        result['time'] = time_result
        torch.save(result, f'/content/result/{basename}_{count}.pt')
        print('==================================================================')
        print('====================== Saving in %d s ============================'%count)
        print('==================================================================')
      keys = data_img.keys()
      frame, dets = detect_obj(model, stride, names, img_detect = frame)
      img = frame
      h, w = img.shape[:2]
      dets = choose_class(names, dets, ["person"])
      for i in dets:
        x1, y1, x2, y2 = list(map(int,i[:4]))
      dets = np.array(dets)
      if not len(dets):
        continue
      # trackers = mot_tracker.update(dets)
      xywhs = torch.from_numpy(xyxy2xywh(dets[:, 0:4]))
      confs = torch.from_numpy(dets[:, 4])
      clss = torch.from_numpy(dets[:, 5])
      dts = torch.from_numpy(dets[:, :4])
      # trackers = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
      # trackers = mot_tracker.update(frame, dets = dts)
      trackers = mot_tracker.update(dets)
      # print(trackers)
      for i in range(len(trackers)):
        ids = str(int(trackers[i][4]))
        x1, y1, x2, y2 = list(map(int,trackers[i][:4]))
        x1, y1, x2, y2 = int_0([x1, y1, x2, y2], w, h)
        img_crop = frame[y1:y2, x1:x2]
        x10, y10 = list(map(lambda x: ((x-5) > 0 and x-5 or 0), [x1, y1]))
        y20 = y2 + 5 > h and h or y2+5
        x20 = x2 + 5 > w and w or x2+5
        x10, y10, x20, y20 = x1, y1, x2, y2
        img_crop = frame[y10:y20, x10:x20]
        if ids not in keys:
          path_save = os.path.join(dir_save, ids)
          data_img[ids] = [0, path_save, []]
          time_result[ids] = count
          if not os.path.isdir(os.path.join(dir_save, ids)):
            print(os.path.join(dir_save, ids))
            os.mkdir(os.path.join(dir_save, ids))
        data_img[ids][0] = data_img[ids][0] + 1
        data_img[ids].append([x10, y10])
        cv2.imwrite(os.path.join(data_img[ids][1], f'{data_img[ids][0]}.jpg'), img_crop)
    else:
      break
  backgroundModels = []
  # print(processing(att_all, label))