import face_model
import argparse
import cv2
import sys
import numpy as np
import json, ast
import codecs

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='Image size in pixels: WxH')
# parser.add_argument('--model', default='/home/ti/Downloads/insightface/deploy/models/model-r100-ii/model,0', help='path to load model.') 
parser.add_argument('--client-features', default='', help='Give the extracted 512d features from client')
parser.add_argument('--server-features', default='', help='Give the extracted 512d features from server')
parser.add_argument('--model', default='/home/ti/Downloads/insightface/recognition/models/kaz/r100-arcface-emore/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()
client_f = args.client_features
server_f = args.server_features
print(client_f)
print(server_f)
# Loading feature from json file - client side
try:
	file = codecs.open(client_f, 'r', encoding='utf-8').read()
	data0 = json.loads(file)
	client = np.array(data0['person'][0]['features'])
except:
	print('Cannot open specified file, please check the path')
# Loading feature from json file - server side
# This part have to be checked and changed according to our
# server architecture and how we gonna take features from DB
try:
	file = codecs.open(server_f, 'r', encoding='utf-8').read()
	data1 = json.loads(file)
	server = np.array(data1['person'][0]['features'])
except:
	print('Cannot open specified file, please check the path')

# Loading model
model = face_model.FaceModel(args)

# Calculating distance
dist = np.sum(np.square(client - server))
print(dist)
sim = np.dot(client, server.T)
print(sim)
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
