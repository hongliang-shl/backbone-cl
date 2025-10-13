#!/usr/bin/env python
# encoding: utf-8

import onnx
import torch
import numpy as np
from PIL import Image
import onnxruntime as ort
from facenet import Facenet

def preprocess_input(image):
    image /= 255.0
    return image

image = Image.open('崇宁通宝_矮示_12.jpg_coin.jpg_160x160.jpg')

model_pytorch = Facenet()
query_fea = model_pytorch.extract_imagefea(image)
print(query_fea)

model_onnx = onnx.load('ep500-loss0.002-val_loss0.912_v2.onnx')
onnx.checker.check_model(model_onnx)

session = ort.InferenceSession('ep500-loss0.002-val_loss0.912_v2.onnx',providers=['CPUExecutionProvider'])

#x=np.random.randn(1,3,160,160).astype(np.float32)

photo = np.expand_dims(np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1)), 0)
outputs = session.run(None, { 'inputs' : photo })

print(outputs)

#np.testing.assert_almost_equal(query_fea[0],outputs[0],decimal=2)
