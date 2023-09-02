import cv2, dlib
import numpy as np
import serial
import time
import os
import winsound as ws
from collections import deque
from imutils import face_utils
from keras.models import load_model

#변수를 넣어서 10회까지만 연속 스택, 이후에는 스택과 팝을 같이해준다. 그리고 배열을 for문으로 계산해서 합으로 상태 확인
arduino = serial.Serial('COM11',9600)


cap = cv2.VideoCapture(0)
IMG_SIZE = (34, 26)
sleep_state=0
sleep_score=0.0
sleep_stack_l=0.0
sleep_stack_r=0.0
keep = 0
i=0


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = load_model('models/2018_12_17_22_58_35.h5')
print('ready?')
model.summary()

print('start?')
def beepsound():
  freq=4000
  dur=2000
  ws.Beep(freq,dur)
def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]
  return eye_img, eye_rect

print('A')
while True:
  time.sleep(0.05)
  keep=keep+1
  ret, img_ori = cap.read()


  img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.8, fy=0.8)

  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = detector(gray)

  for face in faces:
    
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)

    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

    pred_l = model.predict(eye_input_l)
    pred_r = model.predict(eye_input_r)
    # visualize
    state_l = '%.1f' if pred_l > 0.1 else '%.1f'
    state_r = '%.1f' if pred_r > 0.1 else '%.1f'
    state_l = state_l % pred_l
    state_r = state_r % pred_r
    
    sleep_stack_l=float(state_l)
    sleep_stack_r=float(state_r)
    #sleep score값에 따라 아두이노로 신호 보내기
    if(sleep_stack_l<0.1 or sleep_stack_r<0.1):
      sleep_score=sleep_score+1

      
    sleep_score=sleep_score-0.4
    if(sleep_score<0):
      sleep_score=0
    if(sleep_score>10):
      print(beepsound())
      arduino.write(b'y')
      print("Activate")
      sleep_score=0

    

    
    print("sleep score : %f" %sleep_score)
    cv2.rectangle(img,pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2) #왼쪽 눈 박스
    cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2) #오른쪽 눈 박스


#cv2.puttext(그림, 메시지, 포인트, 폰트, 크기, 색깔, 굵기)
    cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2) #왼쪽 눈 점수
    cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2) #오른쪽 눈 점수
  if(keep>60):
    arduino.write(b'k')
    keep=0
  print(keep)
  cv2.imshow('result', img)
  if cv2.waitKey(1) == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()

print('END')
