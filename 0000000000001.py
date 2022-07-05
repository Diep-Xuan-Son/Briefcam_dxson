import cv2


cap = cv2.VideoCapture("./video_test/test2.mp4")
total_frame = int(cap.get(7))
print(total_frame)