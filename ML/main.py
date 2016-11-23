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

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
app.secret_key = os.urandom(24).encode('hex')
CORS(app)

# Set Clarifai API credentials
os.environ['CLARIFAI_APP_ID'] = CLARIFAI_APP_ID
os.environ['CLARIFAI_APP_SECRET'] = CLARIFAI_APP_SECRET

clarifai_api = ClarifaiApi()

# Convert image to vector
def image_vector(img_file):
    img_data = clarifai_api.tag_images(img_file)
    tags = img_data['results'][0]['result']['tag']['classes']
    weights = img_data['results'][0]['result']['tag']['probs']

    vector = [0] * session['current_index']

    for i in range(len(tags)):
        if tags[i] in session['tag_indices']:
            vector[session['tag_indices'][tags[i]]] = weights[i]

    return vector

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/register-account', methods=['GET', 'POST'])
def register_account():
    session['model'] = None
    session['recurring'] = []
    session['tag_indices'] = {}
    session['reverse_tag_indices'] = []
    session['current_index'] = 0
    session['pictures'] = []

    access_token = request.json['token']
    user_id = request.json['user_id']

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

    # Convert all images to vectors
    for media in recent_media:
        img_data = clarifai_api.tag_image_urls(media.images['standard_resolution'].url)
        tags.append(img_data['results'][0]['result']['tag']['classes'])
        weights.append(img_data['results'][0]['result']['tag']['probs'])
        session['recurring'].append(img_data['results'][0]['result']['tag']['classes'])
        date = int(media.created_time.strftime("%s")) * 1000
        dates.append(date)
        time_of_day.append(media.created_time.hour)
        num_likes.append(media.like_count)
        session['pictures'].append([media.images['standard_resolution'].url, img_data['results'][0]['result']['tag']['classes']])

    for vector in tags:
        for tag in vector:
            if tag not in session['tag_indices']:
                session['tag_indices'][tag] = session['current_index']
                session['reverse_tag_indices'].append(tag)
                session['current_index'] += 1

    data = []

    # Generate vectors for each image by marking each tag with weight
    for i in range(len(tags)):
        vector = [0] * session['current_index']

        for j in range(len(tags[i])):
            vector[session['tag_indices'][tags[i][j]]] = weights[i][j]

        # Append extra variables and number of likes
        vector.append(dates[i])
        vector.append(time_of_day[i])
        vector.append(num_likes[i])

        data.append(vector)

    session['model'] = LikePredictor(data)
    return 'Done'

@app.route('/tags')
def tags():
    # Compute most important tags in user's pictures
    important_tags = dict(enumerate(session['model'].regressor.feature_importances_))
    sorted_tags = Counter(important_tags)
    top_ten_tags = []

    for index, importance in sorted_tags.most_common(10):
        # Ensure feature is an image tag
        if index < session['current_index']:
            tag = session['reverse_tag_indices'][index]
            top_ten_tags.append([tag, importance])
    return make_response(json.dumps({'recurring': session['recurring'], 'topten': top_ten_tags, 'pictures': session['pictures']}), 200)

@app.route('/process-image', methods=['POST'])
def process_image():
    vector = image_vector(request.files['image'])

    data = {'prediction': session['model'].predict(vector)}
    response = make_response(json.dumps(data), 200)
    response.headers['Content-Type'] = 'application/json'
    return response

if __name__ == '__main__':
    app.run()
