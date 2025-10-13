#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn
import onnx
from nets.facenet import Facenet as facenet

model = facenet(backbone='inception_resnetv1', mode="predict").eval()
model.load_state_dict(torch.load('logs/1124model/ep397-loss0.003-val_loss0.463.pth', map_location='cpu'), strict=False)

input_names = ['inputs']
output_names = ['outputs']

x = torch.randn(1,3,320,320,requires_grad=True)

torch.onnx.export(model, x, 'recognizer_320x320_v5_1127.onnx', input_names=input_names, output_names=output_names, verbose='True')

