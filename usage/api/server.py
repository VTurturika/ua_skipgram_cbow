from flask import Flask, request, jsonify
from flask_cors import CORS
from models.cbow import get_nearest_cbow
from models.skipgram import get_nearest_skipgram

app = Flask(__name__)
CORS(app)

@app.route('/get-similar', methods=['post'])
def get_similar():
	word = request.get_json()['word']
	cbow = get_nearest_cbow(word)
	skipgram = get_nearest_skipgram(word)
	return jsonify({'word': word, 'cbow': cbow, 'skipgram': skipgram})