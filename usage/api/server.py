from flask import Flask, request
from models.cbow import get_nearest_cbow
from models.skipgram import get_nearest_skipgram

app = Flask(__name__)

@app.route('/<word>')
def hello_world_cbow(word):
	cbow = get_nearest_cbow(word)
	skipgram = get_nearest_skipgram(word)
	return 'CBOW: {0}<br>Skipgram: {1}'.format(cbow, skipgram)