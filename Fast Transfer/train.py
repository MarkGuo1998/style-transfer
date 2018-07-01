from FastTransfer import FastTransfer
from PIL import Image
import numpy as np
import os
import random

style_img = Image.open("./starry-night.jpg")
# content_img = Image.open("tubingen.jpg")
style_img = np.array(style_img.resize((256, 256))).astype('float32')
# content_img = np.array(content_img.resize((256, 256))).astype('float32')
content_layer = {'relu2_2': 1}
style_layer = {'relu1_2': 0.25, 'relu2_2': 0.25, 'relu3_3': 0.25, 'relu4_3': 0.25}

model = FastTransfer(content_layers=content_layer, style_layers=style_layer, 
					 style_image=style_img, lambda_content=0.05, lambda_style=1, 
					 lambda_tv=1e-5, print_loss=100, path='./model/', learning_rate=1e-3, restore_flag=0)

# model.update(learning_rate=0.1, content_image=content_img)

for root, dir, files in os.walk('./val2017'):
	for i in range(40000):
	    file = random.sample(files, 4)
	    for image_name in file:
	        img = Image.open(os.path.join(root, image_name))
	        img = np.array(img.resize((256, 256))).astype('float32')
	        # print(img.shape)
	        model.update(content_image=img)
	    

