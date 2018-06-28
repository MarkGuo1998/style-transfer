from StyleTransfer import StyleTransfer
from PIL import Image
import numpy as np

content = Image.open('./tubingen.jpg')
style = Image.open('./starry-night.jpg')
content = np.array(content.resize((256, 256))).astype('float32')
style = np.array(style.resize((256, 256))).astype('float32')
content_mean = np.mean(content, axis=(0, 1))
style_mean = np.mean(style, axis=(0, 1))
init_image = Image.open('./output/4000.jpg')
init_image = np.array(init_image).astype('float32')


content_layers = {'conv4_2': 1}
style_layers = {'conv1_1': 1 / 5, 'conv2_1': 1 / 5, 'conv3_1': 1 / 5, 'conv4_1': 1 / 5, 'conv5_1': 1 / 5}

style_transfer = StyleTransfer(content_layers=content_layers, style_layers=style_layers, 
                               content_image=content, style_image=style, loss_ratio=0.05, 
                               num_iter=10001, init_image=init_image)   

        
style_transfer.update(learning_rate=0.11, decay_step=200, decay_rate=0.9)
        