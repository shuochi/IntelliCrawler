from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import threading
import re
import gensim
from keras import backend
from main import Main

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

collect = []
path = 'D:\Word2Vec\GoogleNews-vectors-negative300.bin'
W2V = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
print('Model loaded.')

@app.route('/')
def hello_world():
    return 'Focused Crawler!'

@app.route('/start_', methods =['GET', 'POST'])
@cross_origin(origin='*')
def start():
    global collect, W2V
    collect = []
    topics = request.form['topic']
    seeds = request.form['url']
    limit_pages = request.form['page']
    query = re.split(' |,', topics)
    backend.clear_session()
    thread = threading.Thread(target=Main([query, [seeds], \
        int(limit_pages)], W2V, collect))
    thread.start()
    return 'Get to work'

@app.route('/update_', methods =['GET', 'POST'])
@cross_origin(origin='*')
def update():
    global collect
    return jsonify(collect)
