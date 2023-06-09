import pickle
from ctypes import util
from wsgiref.simple_server import make_server

import cv2
import flask
import numpy as np
import werkzeug
from PIL import Image
from flask import send_from_directory, request, render_template, jsonify
import os
app = flask.Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

dic = ['Gram Negative', 'Gram Positive']


@app.route('/', methods=['GET', 'POST'])
def welcome():
    return "Welcome"


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico',
                               mimetype='image/vnd.microsoft.icon')


@app.route('/predict/', methods=['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image0']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = Image.fromarray(img, 'L')
    img = img.resize((128, 128))
    img = np.array(img)
    print("Earlier: ", img.shape)
    img = img.flatten()
    img = img.reshape(1, -1)
    print("Later: ", img.shape)
    predicted_label = model.predict(img)
    # predicted_label = np.argmax(model.predict(np.array([img]))[0], axis=1)
    print(predicted_label)
    return dic[predicted_label[0]]


if __name__ == '__main__':
    # app.run(host="127.0.0.1", port=os.environ.get('PORT', 5000), debug=True)
    app.run(host="0.0.0.0", port=os.environ.get('PORT', 8080), debug=True)
