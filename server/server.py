from flask import Flask, request
from model import get_nearest
app = Flask(__name__)

@app.route('/<word>')
def hello_world(word):
    return get_nearest(word)
