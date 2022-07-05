import os
import numpy as np
import torch
import glob
import cv2
import argparse
from torchvision import transforms as T
from PIL import Image, ImageDraw
def processing(dic, labels):
  label_query = dict()
  for label in labels:
    label_query[label] = []
  for ids in dic:
    attribute = dic[ids]
    len_check = attribute.shape[0]
    if len_check == 0:
      continue
    sum_att = sum(attribute)/len_check
    query = torch.where(sum_att > 0.5,sum_att, torch.tensor(0.).cuda())   
    query = torch.where(query)[0].tolist()
    for index in query:
      label_query[labels[index]].append(ids)
  return label_query
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    parser.add_argument("--weights", default = 'yolov5s.pt', help="Pretrained model detection", type=str)
    parser.add_argument("--use_id", default = False, help="Pretrained model detection", type=bool)
    parser.add_argument("--backbone", default = 'resnet152' , help="Pretrained model detection", type=str)
    parser.add_argument("--input", help="Pretrained model detection", type=str)
    parser.add_argument("--output", default = "result", help="Pretrained model detection", type=str)
    parser.add_argument("--hop", help="Số bước nhảy frame", type=int, default=1)
    args = parser.parse_args()
    return args
def load_network(network):
    save_path = os.path.join('./weights', 'person_mobilenet_v2.pth')
    network.load_state_dict(torch.load(save_path))
    print('Resume model from {}'.format(save_path))
    return network
def extract(img, models, transforms):
    try:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      src = Image.fromarray(img)
      src = transforms(src)
      if torch.cuda.is_available():
        src = src.unsqueeze(dim=0).cuda()
      else:
        src = src.unsqueeze(dim=0)
      out = models.forward(src)
      pred = torch.gt(out, torch.ones_like(out)*0.5 ).int()	# threshold=0.5
      return pred
    except:
      return None
def choose_class(names, lst_array, classes):
  arr = []
  index_clas = []
  for clas in classes:
    index_clas.append(names.index(clas))
  for lst in lst_array:
    if int(lst[5]) in index_clas:
      arr.append(lst)
  return arr