from FastTransfer import FastTransfer
from PIL import Image
import numpy as np

style_img = Image.open("starry-night.jpg")
content_img = Image.open("tubingen.jpg")
style_img = np.array(style_img.resize((256, 256))).astype('float32')
content_img = np.array(content_img.resize((256, 256))).astype('float32')
content_layer = {'relu2_2': 1}
style_layer = {'relu1_2': 0.25, 'relu2_2': 0.25, 'relu3_3': 0.25, 'relu4_3': 0.25}

model = FastTransfer(content_layers=content_layer, style_layers=style_layer, 
                     content_image=content_img, style_image=style_img,
                     lambda_content=0.05, lambda_style=1, lambda_tv=1e-5, num_iter=1000)

model.update(learning_rate=0.1)