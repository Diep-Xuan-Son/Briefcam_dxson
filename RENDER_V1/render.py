import cv2
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import os
import logging
import argparse
import json
from zipfile import ZipFile
sys.path.append("/home/son/AI/Briefcam/BRIEFCAM_V2")
from utils.plots import plot_one_box

def plot(x1, y1, x2, y2, time, img_background, color):
  img0 = np.asarray(img_background)
  xyxy = [x1, y1, x2, y2]
  label = f'{time}'
  plot_one_box(xyxy, img0, label=label, color=color , line_thickness=1)
  # img_background = Image.fromarray(img0)
  return img_background

def iou(bbox1, bbox2):
  """
  Calculates the intersection-over-union of two bounding boxes.
  Args:
    bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
  Returns:
    int: intersection-over-onion of bbox1, bbox2
  # """
  bbox1 = [float(x) for x in bbox1]
  bbox2 = [float(x) for x in bbox2]
  (y0_1, x0_1, y1_1, x1_1) = bbox1
  (y0_2, x0_2, y1_2, x1_2) = bbox2
  # get the overlap rectangle
  overlap_x0 = max(x0_1, x0_2)
  overlap_y0 = max(y0_1, y0_2)
  overlap_x1 = min(x1_1, x1_2)
  overlap_y1 = min(y1_1, y1_2)
  # check if there is an overlap
  if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
    return 0
  # if yes, calculate the ratio of the overlap to each ROI size and the unified size
  size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
  size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
  size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
  size_union = size_1 + size_2 - size_intersection
  return size_intersection / size_union
def check_iou(bbox, bboxs):
  for bb in bboxs:
    if iou(bbox, bb) > 0 and iou(bbox, bb) != 1:
      return True
  return False
def paste_v2(bgs1, frames, bboxs, time_obj, blue=5, color=[255,255,255]):  #bgs1:background dang xet, frames:list cac anh obj dang xet, bboxs:toa do, time_obj:thoi gian voi ids tuong ung, blue:so nguyen co dinh de noi rong box
  # font           = cv2.FONT_HERSHEY_SIMPLEX
  # fontColor       = (255,0,0) #BGR
  # thickness       = 1
  # lineType         = 1
  bgs1 = Image.fromarray(cv2.cvtColor(bgs1, cv2.COLOR_BGR2RGB))
  ## mask = Image.new(mode="L", size=bgs1.size)
  bgs = bgs1.copy()
  ## draw = ImageDraw.Draw(mask)
  k = max(bgs.size[1] , bgs.size[0])
  fontScale = (bgs.size[1] * bgs.size[0]) / (k**2.14)
  # print(k**2,bgs.size[1] * bgs.size[0] )
  for inter, img_rd in enumerate(frames):
    # mask = Image.new(mode="L", size=bgs1.size)
    # draw = ImageDraw.Draw(mask)
    # x1, y1, x2, y2 = bboxs[inter]
    # a = x2-x1 # w
    # b = y2-y1 # h
    # img_rd = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
    # xyxy = bboxs[inter]
    # if check_iou(xyxy, bboxs): #neu xyxy dang xet bi overlap len mot box trong bboxs thi se bi lam mo
    #   draw.rectangle([x1 + blue, y1+blue, x2-blue, y2-blue], fill=175, outline=None)
    # else:
    #   draw.rectangle([x1 + blue, y1+blue, x2-blue, y2-blue], fill=255, outline=None)
    # mask_crop = mask.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(3))
    # # mask_crop.show()
    # # if check_iou(xyxy, bboxs):
    # #   mask_crop = mask.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(3))
    # bgs.paste(img_rd, (int(x1), int(y1)), mask_crop)
    #----------------------------------------------------
    x1, y1, x2, y2 = bboxs[inter]
    xyxy = bboxs[inter]
    img_rd = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
    if check_iou(xyxy, bboxs): #neu xyxy dang xet bi overlap len mot box trong bboxs thi se bi lam mo
      img_rd.putalpha(80)
      bgs.paste(img_rd, (int(x1), int(y1)), img_rd)
    else:
      bgs.paste(img_rd, (int(x1), int(y1)))
  opencv_image = cv2.cvtColor(np.asarray(bgs), cv2.COLOR_RGB2BGR)
  for inter, img_rd in enumerate(frames):
    x1, y1, x2, y2 = bboxs[inter]
    opencv_image = plot(x1, y1, x2, y2, time_obj[inter][:8], opencv_image, color=color)
    # opencv_image = cv2.putText(opencv_image , time_obj[inter][:8], (x1, y1),font, fontScale, fontColor, thickness, lineType)
  return opencv_image

def creat_bgs(name_video, path_bgs):
  cap = cv2.VideoCapture(name_video)
  count = 0
  images = []
  while True:
    ret, frame = cap.read()
    if ret:
      count += 1
      images.append(frame)
      if count % 200 == 0:
      # if count % (total_frame*10//100) == 0:
        if count % 2000 == 0:
        # if count % (total_frame*20//100) == 0:
          bgs_pth = os.path.join(path_bgs, f"bgs_{count}.jpg")
          cv2.imwrite(bgs_pth, np.median(images, axis = 0))
        images.clear()
    else:
      bgs_pth = os.path.join(path_bgs, f"bgs_{2000*(1+count//2000)}.jpg")
      cv2.imwrite(bgs_pth, np.median(images, axis = 0))
      break
  cap.release()

def chooes_bgs(val, my_dict, path_img, percent_total_frame):
  anh_nen = ''

  if(len(os.listdir(path_img)) != 0):
    current_frame_count = my_dict[val][0] * framerate / hop

    # if( int(current_frame_count//2000) <=  int(frame_total//2000) - 1 ):
    if( int(current_frame_count//percent_total_frame) <=  int(frame_total//percent_total_frame) - 1 ):
      # STT_anh_nen = 2000 + 2000 * int(current_frame_count//2000)
      STT_anh_nen = percent_total_frame + percent_total_frame * int(current_frame_count//percent_total_frame)
    else:
      # STT_anh_nen = 2000 * int(current_frame_count//2000)
      STT_anh_nen = percent_total_frame * int(current_frame_count//percent_total_frame)

    link_anh_nen = os.path.join(path_img, f"bgs_{STT_anh_nen}.jpg")
    anh_nen = cv2.imread(link_anh_nen)

  else:
    print("Don't have background image")

  return anh_nen

def compute_date(time_):
  gio = time_//3600
  phut = time_ // 60
  giay = time_  % 60

  sec = str(int(giay)).rjust(2, '0')
  minute = str(int(phut)).rjust(2, '0')
  hour = str(int(gio)).rjust(2, '0')

  return f'{hour}:{minute}:{sec}'
def check_all(arr_render, img):
  for i in arr_render:
    if len(img[i]) > 3:
      return True
  return False

def parse_args():   # phuong thêm
  """Parse input arguments.""" 
  parser = argparse.ArgumentParser(description='render')
  parser.add_argument("--data_input", help="", type=str)
  parser.add_argument("--video_output_path", help="", type=str)
  args = parser.parse_args()
  return args

    
if __name__ == "__main__":
  args = parse_args()   # phuong thêm
  logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

  #load thông số file video
  vid = ''
  frame_width = 0
  frame_height = 0
  framerate = 0
  hop = 0
  frame_total = 0
  with open(f'{args.data_input}/video_para.json', 'r') as labels:  #f'{path_save}/{vid}.pt's
      thongsovideo = json.load(labels)

      vid = thongsovideo['name']
      framerate = thongsovideo['fps']
      frame_width = thongsovideo['width']
      frame_height = thongsovideo['height']
      hop = thongsovideo['hop']
      frame_total = thongsovideo['frame_total']
      print("thông số video là:", thongsovideo)

  path_save = args.data_input # "/content/drive/MyDrive/phuong/ket_qua2" # Vị trí folder lưu các đối tượng và data
  video_output_path = args.video_output_path 
  path_bgs = f"{path_save}/{vid}_bgs"
  data = torch.load(f'{path_save}/{vid}.pt')
  print("Phương comment phục vụ render cho TraDeS") #att = data['att']
  img = data['data'] # data ảnh
  time_rs = data['time']  # key:ids; value:list of time ids appear
  key = []
#ifdef 1:
  att_query = ['test']
  # for i in att:
  #   if len(att[i]) != 0:
  #     print(i)
  # att_query = input('Nhap vao thuoc tinh can theo doi: ')
  # att_query = list(map(str,att_query.split(' ')))
  # print("các thuộc tính tìm kiếm là:", att_query)

  # # if not len(att_query): # sau tich hop se dung den

  # for thuoctinh in att_query:
  #   key += att[thuoctinh]  
  # print("các ID là:", key)

  # if len(key):
  #   a = list(time_rs.keys())
  #   print("các a la:", a)
  #   for ke in a:
  #     if ke not in key:
  #       del time_rs[ke]
#endif
  print("tiếp tục chạy")
  sort = sorted(time_rs.items() , key = lambda item: item[1]) # link https://stackoverflow.com/questions/8966538/syntax-behind-sortedkey-lambda #item[1]: sort time_rs theo list time
  count_frame = dict()

  out = cv2.VideoWriter(os.path.join(video_output_path,f'{vid}_{att_query}_brief.avi'),cv2.VideoWriter_fourcc(*'mp4v'), framerate, (frame_width, frame_height))
  for key, inter in sort:
    if inter[0] not in count_frame.keys():
      count_frame[inter[0]] = []
      count_frame[inter[0]].append(key)   #chuyen tu dict ids:time => time:ids (time la thoi gian dau tien object do xuat hien)
    else:
      count_frame[inter[0]].append(key)
  arr_render = []
  numframerender = 0
  npercent_total_frame = (frame_total*20//100) #them
  keys = list(count_frame.keys()) #list time dau tien
  number_img = dict()

  color = [np.random.randint(0, 255) for _ in range(3)]
  while True:
    if not len(arr_render):
      arr_render = count_frame[keys[0]] #list ids cua 1 time dau tien
      keys.pop(0)
    bgs = chooes_bgs(arr_render[-1], time_rs, path_bgs, npercent_total_frame)  #them  meaning: neu ids dang xet co thoi gian xuat hien dau tien trong khoang 20% thoi gian nao do cua video thi se lay background trong khoang thoi gian do
    frames = []
    bboxs = []
    time_obj = []
    # Check xem cứ 20 giây thì sẽ cho xe mới vào
    if numframerender % 10 == 0 and len(keys) != 0: # cứ sau khi render được 10 frame thì sẽ đưa xe tiếp theo vào render
      arr_render += count_frame[keys[0]]
      keys.pop(0)
    # check xem nếu id xe nào hết ảnh render thì dồn tiếp id xe tiếp theo
    if not check_all(arr_render, img) and len(keys):
      arr_render += count_frame[keys[0]]
      keys.pop(0)
    for idd in arr_render:
        if len(img[idd]) == 3:
          arr_render.remove(idd)
    #Render
    # print(arr_render)
    # exit()
    for i in arr_render:
      # for idd in arr_render:
      #   if len(img[idd]) == 3:
      #     arr_render.remove(idd)
      if i not in number_img.keys():
        number_img[i] = 0 #stt object cua tung folder ids
      root =  os.path.join(path_save,"/".join(img[i][1].split("/")[-2:]))
      path = os.path.join(root, f"{number_img[i]}.jpg")
      img_render = cv2.imread(path)
      x0, y0 = img[i][3]
      bbox = [x0, y0, x0+img_render.shape[1], y0+img_render.shape[0]]
      frames.append(img_render)
      bboxs.append(bbox)
      # time_obj.append(str(i))
      time_obj.append(compute_date(time_rs[i][number_img[i]] )) # phuong sua # goc time_obj.append(compute_date(time_rs[i]/framerate))
      number_img[i] += 1
      img[i].pop(3)
    numframerender += 1
    frame = paste_v2(bgs, frames, bboxs, time_obj, blue=5, color=color)
    if not len(arr_render):
      break
    out.write(frame)
    # cv2.imshow("1", frame)
    cv2.waitKey(1)
  out.release()

  print("Đã render xong")



# python3 ./RENDER_V1/render.py --data_input ./result --video_output_path ./video_brief
