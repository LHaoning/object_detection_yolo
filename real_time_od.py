# -*- coding: utf-8 -*-
"""real_time_od.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Bbd0elmzLiNAlih_qQf9v0cup7Dp1Z7q

# **Import libraries**
"""

import cv2
import time
from keras import backend as K
from yolo import video_predict

"""**Initialize the video stream**"""

cap = cv2.VideoCapture(0)
time.sleep(2)

"""**Loop over the frames from the video stream**"""

model_image_size = (608, 608)
while True:
  # grab the frame from the video stream and reshape image size
  ret, img = cap.read()
  img = cv2.resize(img, model_image_size)
  
  # Run the graph on the frame from the video stream
  sess = K.get_session()
  img = video_predict(sess, img)
  cv2.imshow('od', img)

  key = cv2.waitKey(10)
  if key == 27:
      break

"""**Release the video stream**"""

cv2.destroyAllWindows()
cv2.VideoCapture(0).release()