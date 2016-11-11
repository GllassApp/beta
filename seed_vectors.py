import json
from config import *
from mongoengine import *
from models import *

connect(MONGODB_NAME, host=MONGODB_URI)

with open('glove/glove.6B.50d.txt') as f:
    for line in f:
        split_line = line.split()
        word = split_line[0]

        vector = []

        for i in range(1, len(split_line)):
            vector.append(float(split_line[i]))

        word_vector = WordVector(word=word, vector=vector)
        word_vector.save()