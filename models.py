from mongoengine import *

class WordVector(Document):
    word = StringField(required=True)
    vector = ListField(required=True)