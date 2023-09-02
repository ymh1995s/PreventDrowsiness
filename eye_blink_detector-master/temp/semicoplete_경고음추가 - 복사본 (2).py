import cv2, dlib
import numpy as np
import serial
import time
import os
import winsound as ws

import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout, QApplication
from PyQt5.QtCore import QObject
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtCore import QThread
from collections import deque
from imutils import face_utils
from keras.models import load_model

#변수를 넣어서 10회까지만 연속 스택, 이후에는 스택과 팝을 같이해준다. 그리고 배열을 for문으로 계산해서 합으로 상태 확인
arduino = serial.Serial('COM9',9600)


cap = cv2.VideoCapture(0)
IMG_SIZE = (34, 26)
sleep_state=0
sleep_score=0.0
sleep_stack_l=0.0
sleep_stack_r=0.0
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
######################## pyqt
#// widget UI setting
class Window(QWidget):

    def __init__(self):
        super().__init__()

        self.button_start = QPushButton('Start', self)
        self.button_cancel = QPushButton('Cancel', self)
        self.label_status = QLabel('status!!', self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.button_start)
        layout.addWidget(self.button_cancel)
        layout.addWidget(self.label_status)

        self.setFixedSize(400, 200)
    @pyqtSlot(int)
    def updateStatus(self, status):
          self.label_status.setText('{}'.format(status))


#// main loop
class Example(QObject):

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)

        self.gui = Window()
        self.worker = Worker()               # 백그라운드에서 돌아갈 인스턴스 소환
        self.worker_thread = QThread()       # 따로 돌아갈 thread를 하나 생성
        self.worker.moveToThread(self.worker_thread)# worker를 만들어둔 쓰레드에 넣어줍니다
        self.worker_thread.start()           # 쓰레드를 실행합니다.

        self._connectSignals()               # 시그널을 연결하기 위한 함수를 호출


        self.gui.show()
    def _connectSignals(self):
        self.gui.button_start.clicked.connect(self.worker.startWork)
        self.worker.sig_numbers.connect(self.gui.updateStatus)
        self.gui.button_cancel.clicked.connect(self.forceWorkerReset)


    def forceWorkerReset(self):
        if self.worker_thread.isRunning():  #// 쓰레드가 돌아가고 있다면 
            self.worker_thread.terminate()  #// 현재 돌아가는 thread 를 중지시킨다
            self.worker_thread.wait()       #// 새롭게 thread를 대기한후
            self.worker_thread.start()      #// 다시 처음부터 시작




class Worker(QObject):
    sig_numbers = pyqtSignal(int)

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)

    @pyqtSlot()          # // 버튼이 클릭시 시그널을 받아들이는 슬롯을 하나 마련합니다. 
    def startWork(self):
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
      if(sleep_stack_l<0.1 and sleep_stack_r<0.1):
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

      cv2.imshow('result', img)
        ####################################end pyqt
print('A')

if __name__ == "__main__":
  app = QApplication(sys.argv)
  example = Example(app)
  sys.exit(app.exec_())

  
cap.release()
cv2.destroyAllWindows()

print('END')
