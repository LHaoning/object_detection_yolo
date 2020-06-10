# -*- coding: utf-8 -*-
"""image_od

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bYWenJ9BGhpIs7HgljiKhs0lX9_hzv73

# **Import libraries**
"""

from keras import backend as K
from yolo import predict

"""**Defining image path**"""

image_file = 'test_001.jpg'

"""**Run the graph on an image**"""

sess = K.get_session()
out_scores, out_boxes, out_classes = predict(sess, image_file)