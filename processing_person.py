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
from net.models import Backbone_Mobilenet
# from deep_sort_pytorch.deep_sort import DeepSort
from utils.general import xyxy2xywh
from utils.downloads import attempt_download
from Sort.sort import *
from collections import defaultdict   # phuong them , tao list cho dict, lam theo link   https://www.geeksforgeeks.org/python-initializing-dictionary-with-empty-lists/
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
  save_path = os.path.join('./weights', 'person_mobilenet_tru_nen_xanh_v1.pth')  # person_mobilenet_cu.pth  person_mobilenet_tru_nen_xanh_v1  ko_tru_nen_v1
  print("Duong dan model thuoc tinh la:", save_path)
  network.load_state_dict(torch.load(save_path))
  print('Resume model from {}'.format(save_path))
  return network

if __name__ == '__main__':
  result = dict()
  data_img = dict()
  att_all = dict()
  time_result = defaultdict(list) # cu: time_result = dict()
  #Model nhận diện thuộc tính
  with open('./doc/data_duke_market_seg_v1.json', 'r') as labels:   # person_final_label_cu.json  data_duke_market_seg_v1
      # label = json.load(labels)
      # label= list(label.values())  # json và weight cũ: day la json format dict, lay ra value
      label = json.load(labels)["label"]   # ap dung cho json label format moi
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
  model_att = Backbone_Mobilenet(num_label, model_name='mobilenet_v2')
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
  dir_save = f'{args.output}/{basename}'
  path_bgs = f"{args.output}/{basename}_bgs"
  if not os.path.isdir(dir_save):
    os.mkdir(dir_save)
  if not os.path.isdir(path_bgs):
    os.mkdir(path_bgs)
  backgroundModels = []
  time_ = 0
  fps = cap.get(5)
  video_width = int(cap.get(3))
  video_height = int(cap.get(4))
  total_frame = int(cap.get(7))
  print("fps của video gốc là:", fps)

  thongsovideo = dict()

  thoigian_sort_total = 0
  thoigian_extract_total = 0
  thoigian_tao_folder_total = 0
  thoigian_detect_total = 0
  thoigian_loc_person_total = 0
  thoigian_doc_video_total = 0

  # print("Phuong fix class = 0 ")

  t1 = time.time()
  thoigian_chaychuongtrinh_begin = time.time()

  images = []
  bgs_200 = []
  curr_frame = 0
  count_after_20percent = 0
  hop = args.hop  # phuong them
  while True:
    thoigian_doc_video_begin = time.time()
    ret, frame = cap.read()
    thoigian_doc_video_total += ( time.time() - thoigian_doc_video_begin)
    curr_frame += 1
    
    if not ret: break
    if curr_frame % hop == 0:
      count += 1   
      images.append(frame)
      # if count % 200 == 0:
      if count % (total_frame*10//100) == 0:
        bgs_200.append(np.median(images, axis = 0))
        # if count % 2000 == 0:
        if count % (total_frame*20//100) == 0:
          bgs_pth = os.path.join(path_bgs, f"bgs_{count}.jpg")
          cv2.imwrite(bgs_pth, np.median(bgs_200, axis = 0))
          bgs_200.clear()
        images.clear()
      time_ = ( count*hop )/fps # Tính thời gian   # phuong them hop
      # if count % 100 == 0:
      if count % (total_frame*20//100) == 0:
        print(f'Processing in frame number {count} , FPS: {(count-count_after_20percent)/(time.time()-t1)}')
        count_after_20percent = count
        t1 = time.time()
      # if count % 7000 == 0:
      #   result['att'] = processing(att_all, label)
      #   result['data'] = data_img
      #   result['time'] = time_result
      #   torch.save(result, f'{args.output}/{basename}_{count}.pt')
      #   with open(f'{args.output}/{basename}.json', "w") as filejson:
      #     json.dump(result, filejson, indent = 4) 
      #   print('==================================================================')
      #   print('====================== Saving in frame number %d  ============================'%count)
      #   print('==================================================================')
      keys = data_img.keys()

      thoigian_detect_begin = time.time()
      frame, dets = detect_obj(model, stride, names, img_detect = frame)  #dets=[x1, y1, x2, y2, acc, cls], cls: la chi so index tuong ung voi ten class
      thoigian_detect_total += (time.time() - thoigian_detect_begin)
      img = frame
      h, w = img.shape[:2]

      thoigian_loc_person = time.time()
      dets = choose_class(names, dets, ["person"])
      thoigian_loc_person_total += (time.time() - thoigian_loc_person)
      for i in dets:
        x1, y1, x2, y2 = list(map(int,i[:4]))
      dets = np.array(dets)
      if not len(dets):
        continue

      xywhs = torch.from_numpy(xyxy2xywh(dets[:, 0:4]))
      confs = torch.from_numpy(dets[:, 4])
      clss = torch.from_numpy(dets[:, 5])
      dts = torch.from_numpy(dets[:, :4])

      thoigian_sort_1 = time.time()
      trackers = mot_tracker.update(dets)
      thoigian_sort_total += (time.time() - thoigian_sort_1) 

      for i in range(len(trackers)):
        ids = str(int(trackers[i][4]))
        x1, y1, x2, y2 = list(map(int,trackers[i][:4]))
        x1, y1, x2, y2 = int_0([x1, y1, x2, y2], w, h)
        img_crop = frame[y1:y2, x1:x2]

        thoigian_extract_begin = time.time()
        pred = extract(img_crop, model_att, transforms) 
        thoigian_extract_total += (time.time() - thoigian_extract_begin)

        x10, y10 = list(map(lambda x: ((x-5) > 0 and x-5 or 0), [x1, y1]))
        y20 = y2 + 5 > h and h or y2+5
        x20 = x2 + 5 > w and w or x2+5
        x10, y10, x20, y20 = x1, y1, x2, y2
        img_crop = frame[y10:y20, x10:x20]

        thoigian_tao_folder_begin = 0

        if ids not in keys:
          att_all[ids] = torch.FloatTensor().cuda() 
          path_save = os.path.join(dir_save, ids)
          data_img[ids] = [0, path_save, []]
          # phuong chuyen xuong duoi time_result[ids] = time_
          thoigian_tao_folder_begin = time.time()
          if not os.path.isdir(os.path.join(dir_save, ids)):
            # print(os.path.join(dir_save, ids))
            os.mkdir(os.path.join(dir_save, ids))
          
          thoigian_tao_folder_total += (time.time() - thoigian_tao_folder_begin)

        data_img[ids].append([x10, y10])

        thoigian_tao_anh = time.time()

        cv2.imwrite(os.path.join(data_img[ids][1], f'{data_img[ids][0]}.jpg'), img_crop)

        thoigian_tao_folder_total += (time.time() - thoigian_tao_anh)

        time_result[ids].append(time_)    # phuong chuyển từ trên xuống, để cập nhật thời gian cho từng frame
        data_img[ids][0] = data_img[ids][0] + 1   # phuong: lũy kế số thứ tự frame
        if (x2-x1) > 50 and (y2 -y1) > 50:  
          att_all[ids] = torch.cat((att_all[ids], pred), 0)   # att_all[ids] = [] => cat => att_all[ids] = [0 1 0 1 0 1 ....] (tuy vao so label)

    

  backgroundModels = []

  thoigian_processing = time.time()
  # print(processing(att_all, label)) 
  result['att'] = processing(att_all, label)
  print("thoi gian processing =", (time.time() - thoigian_processing) )
  result['data'] = data_img     #data_img[ids][1] = path_save = os.path.join(dir_save, ids); data_img[ids][0]: la thu tu anh trong moi thu muc IDS
  result['time'] = time_result

  print("Tong thoi gian chay chuong trinh = ", (time.time() - thoigian_chaychuongtrinh_begin  ))  

  torch.save(result, f'{args.output}/{basename}.pt')
  with open(f'{args.output}/{basename}.json', "w") as filejson:
    json.dump(result, filejson, indent = 4) 

  print("Thoi gian doc video =", thoigian_doc_video_total)
  print("thoi gian detect =", thoigian_detect_total, "thoi gian loc person =", thoigian_loc_person_total)
  print("tong thoi gian sort =", thoigian_sort_total, "thoi gian extract =", thoigian_extract_total, "thoi gian tao folder =", thoigian_tao_folder_total)

  thongsovideo['name'] = basename
  thongsovideo['fps'] = int(fps)
  thongsovideo['width'] = video_width
  thongsovideo['height'] = video_height  
  thongsovideo['hop'] = args.hop
  thongsovideo['frame_total'] = count
  with open(f'{args.output}/video_para.json', "w") as filejson: # with open(f'{args.output}/{basename}_video_para.json', "w") as filejson:
    json.dump(thongsovideo, filejson, indent = 4) 


# rm -rf /content/drive/MyDrive/Son/BriefCam/BRIEFCAM_V2/result
# mkdir /content/drive/MyDrive/Son/BriefCam/BRIEFCAM_V2/result
# python3 processing_person.py --input /content/drive/MyDrive/Son/BriefCam/BRIEFCAM_V2/video_test/video4.mp4 --weights ./weights/yolov5s.pt --output /content/drive/MyDrive/Son/BriefCam/BRIEFCAM_V