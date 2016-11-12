# Convert all images in the images folder to vectors

import os
from config import *
from clarifai.client import ClarifaiApi

# Set Clarifai API credentials
os.environ['CLARIFAI_APP_ID'] = CLARIFAI_APP_ID
os.environ['CLARIFAI_APP_SECRET'] = CLARIFAI_APP_SECRET

clarifai_api = ClarifaiApi()

with open('tag_data.txt', 'w') as f:
    for image in os.listdir('images'):
        img_data = clarifai_api.tag_images(open('images/%s' % image, 'rb'))
        tags = img_data['results'][0]['result']['tag']['classes']

        line = ''

        # Write tag data to file
        for i in range(len(tags)):
            # Use comma as delimiter
            if i is not 0:
                line += ','
            line += tags[i]

        f.write(line + '\n')