from flask import Flask, render_template, request, url_for, redirect, make_response, session
from flask_cors import CORS, cross_origin
from ml import *
import os
from config import *
from clarifai.client import ClarifaiApi
import json
from instagram.client import InstagramAPI
import requests
import datetime
from collections import Counter
from sys import argv
import operator

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
app.secret_key = os.urandom(24).encode('hex')
CORS(app)

# Set Clarifai API credentials
os.environ['CLARIFAI_APP_ID'] = CLARIFAI_APP_ID
os.environ['CLARIFAI_APP_SECRET'] = CLARIFAI_APP_SECRET

clarifai_api = ClarifaiApi()

model = None
recurring = []
tag_indices = {}
reverse_tag_indices = []
current_index = 0
pictures = []

# Convert image to vector
def image_vector(img_file):
    img_data = clarifai_api.tag_images(img_file)
    tags = img_data['results'][0]['result']['tag']['classes']
    weights = img_data['results'][0]['result']['tag']['probs']

    vector = [0] * current_index

    for i in range(len(tags)):
        if tags[i] in tag_indices:
            vector[tag_indices[tags[i]]] = weights[i]

    return vector

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/register-account', methods=['GET', 'POST'])
def register_account():
    access_token = request.json['token']
    user_id = request.json['user_id']
    target = open('tokens.txt', 'a')
    target.write('token: '+access_token+' user_id: '+user_id+'\n')
    if not access_token:
        return 'Missing access token'

    session['access_token'] = access_token
    session['user_id'] = user_id

    api = InstagramAPI(access_token=access_token, client_secret=IG_CLIENT_SECRET)

    recent_media, next_ = api.user_recent_media(user_id=user_id, count=20)

    tags = []
    num_likes = []
    dates = []
    time_of_day = []
    weights = []

    global recurring
    global pictures
    # Convert all images to vectors
    for media in recent_media:
        img_data = clarifai_api.tag_image_urls(media.images['standard_resolution'].url)
        tags.append(img_data['results'][0]['result']['tag']['classes'])
        weights.append(img_data['results'][0]['result']['tag']['probs'])
        recurring.append(img_data['results'][0]['result']['tag']['classes'])
        date = int(media.created_time.strftime("%s")) * 1000
        dates.append(date)
        time_of_day.append(media.created_time.hour)
        num_likes.append(media.like_count)
        pictures.append([media.images['standard_resolution'].url, img_data['results'][0]['result']['tag']['classes']])
    # Dictionary to store indices of tags
    global tag_indices
    # Iterator for indices
    global current_index

    for vector in tags:
        for tag in vector:
            if tag not in tag_indices:
                tag_indices[tag] = current_index
                reverse_tag_indices.append(tag)
                current_index += 1

    data = []

    # Relative importances of tags in pictures
    tag_scores = {}

    # Number of images that tags appear in
    tag_count = {}

    # Generate vectors for each image by marking each tag with weight
    for i in range(len(tags)):
        vector = [0] * current_index

        for j in range(len(tags[i])):
            vector[tag_indices[tags[i][j]]] = weights[i][j]

            if tags[i][j] in tag_scores:
                tag_scores[tags[i][j]] += weights[i][j] * num_likes[i]
                tag_count[tags[i][j]] += 1
            else:
                tag_scores[tags[i][j]] = weights[i][j] * num_likes[i]
                tag_count[tags[i][j]] = 1

        # Append extra variables and number of likes
        vector.append(dates[i])
        vector.append(time_of_day[i])
        vector.append(num_likes[i])

        data.append(vector)

    # Divide tag scores by number of pictures they appear in
    for key, value in tag_scores.iteritems():
        tag_scores[key] /= tag_count[key]
    sorted_tag_scores = sorted(tag_scores.items(), key=operator.itemgetter(1), reverse=True)

    # Top ten most important tags
    top_ten_tags = []
    for elem in sorted_tag_scores[:10]:
        top_ten_tags.append([elem[0], elem[1]])

    global model
    model = LikePredictor(data)

    response =  make_response(json.dumps({'recurring': recurring, 'topten': top_ten_tags, 'pictures': pictures}))
    pictures = []
    recurring = []
    top_ten_tags = []
    sorted_tags = []
    return response

@app.route('/process-image', methods=['POST'])
def process_image():
    vector = image_vector(request.files['image'])

    global model
    global current_index
    global tag_indices

    data = {'prediction': model.predict(vector)}
    response = make_response(json.dumps(data), 200)
    response.headers['Content-Type'] = 'application/json'
    return response

if __name__ == '__main__':
    app.run(debug=True)
