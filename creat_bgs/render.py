import cv2
import numpy as np
frame = []
cap = cv2.VideoCapture('/content/drive/MyDrive/video/videoplayback.webm')
while cap.isOpened():
    ret, frame1 = cap.read()
    frame.append(frame1)
    frame1 = np.median(frame, axis=0).astype(dtype=np.uint8)
    cv2.imwrite('/content/drive/MyDrive/BRIEFCAM/render/bgs.jpg', frame1)
    if len(frame) == 2000:
        break