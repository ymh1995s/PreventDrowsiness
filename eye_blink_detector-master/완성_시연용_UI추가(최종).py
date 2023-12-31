import cv2, dlib
import numpy as np
import serial
import time
import os
import keyboard
import winsound as ws
from collections import deque
from imutils import face_utils
from keras.models import load_model

arduino = serial.Serial('COM11',9600)


cap = cv2.VideoCapture(0)
IMG_SIZE = (34, 26)
sleep_state=0
sleep_score=0.0
sleep_stack_l=0.0
sleep_stack_r=0.0
criteria=0.3

keep = 0
i=0


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = load_model('models/2018_12_17_22_58_35.h5')
print('ready?')
model.summary()

print('start?')
def waring():
  freq=4000
  dur=200
  ws.Beep(freq,dur)
def activate():
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


while True:
  
  time.sleep(0.05)
  if keyboard.is_pressed('u'):
    if(criteria<0.81):
      criteria=criteria+0.1
      criteria=round(criteria,2)
  if keyboard.is_pressed('d'):
    if (criteria>0.11):
      criteria=criteria-0.1
      criteria=round(criteria,2)

  textcriteria="criteria : "
  temp = str(criteria)
  textcriteria+=temp
  facestate=0
  keep=keep+1
  ret, img_ori = cap.read()
  if(sleep_score<6):
    statenum=0
    state="state : Good"
  elif(sleep_score>6 and sleep_score<9):
    statenum=1
    state="state : Waring"
  elif(sleep_score>=9):
    statenum=2
    state="state : Danger"
  onface="face off"

  img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.8, fy=0.8)

  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = detector(gray)

  for face in faces:

    onface="face : on"
    facestate=1
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
    if(sleep_stack_l<criteria and sleep_stack_r<criteria):
      sleep_score=sleep_score+1

      
    sleep_score=sleep_score-0.4
    if(sleep_score<0):
      sleep_score=0
    if(sleep_score>6 and sleep_score<12):
      #print(waring())
      print("졸리신가요?")
    if(sleep_score>12):
      print(activate())
      arduino.write(b'y')
      print("Activate")
      sleep_score=0

    

    
    #print("sleep score : %f" %sleep_score)
    cv2.rectangle(img,pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2) #왼쪽 눈 박스
    cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2) #오른쪽 눈 박스


#cv2.puttext(그림, 메시지, 포인트, 폰트, 크기, 색깔(b,g,r), 굵기)
    cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2) #왼쪽 눈 점수
    cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2) #오른쪽 눈 점수

  round(criteria,2)
  if(keep>60):
    arduino.write(b'k')
    keep=0
    print("절전방지")
  if(statenum==0):
    cv2.putText(img, state, (30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
  elif(statenum==1):
    cv2.putText(img, state, (30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
  elif(statenum==2):
    cv2.putText(img, state, (30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
  
  if(facestate==0):
    cv2.putText(img, onface, (30,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
  elif(facestate==1):
    cv2.putText(img, onface, (30,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


  #if(facestate==0):
    #cv2.putText(img, textcriteria, (30,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
  cv2.putText(img, textcriteria, (30,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

  cv2.imshow('result', img)
  
  if cv2.waitKey(1) == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()

print('END')
