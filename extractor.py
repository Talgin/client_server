import face_model
import argparse
import cv2
import sys
import numpy as np
import json

parser = argparse.ArgumentParser(description='Extracting input image features')
#general
parser.add_argument('--image-size', default='112,112', help='Image size in pixels')
# parser.add_argument('--model', default='/home/ti/Downloads/insightface/deploy/models/model-r100-ii/model,0', help='path to load model.') 
parser.add_argument('--features-file', default='', help='Give the extracted 512d features from client')
parser.add_argument('--image', default='Alibek_Datbayev_0003.png', help='Give the path to an image')
parser.add_argument('--model', default='/home/ti/Downloads/SERVER_CODE/models/kaz/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()
load_image = args.image
feature_file = args.features_file

model = face_model.FaceModel(args)
img = cv2.imread(load_image)
img = model.get_input(img)
f1 = model.get_feature(img)

load_image = load_image.split('.')
name = load_image[0]

ft = f1.tolist()
print(ft)
# jstr = str(f1)
# Writing features to some json file
data = {
    "person": [
    {
        "name": name,
        "features": ft
    }
    ]
}
x = json.dumps(data)
print(x)
try:
	with open(feature_file, 'w') as write_file:
		json.dump(data, write_file)
except:
	print('Could not write to a specified file')

print(f1[0:10])
