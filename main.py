import numpy as np
import flask
import werkzeug
from flask import Flask,request,jsonify
import pickle
import imageio
model = pickle.load(open('X.pickle','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Sam"


@app.route('/predict/', methods=['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image0']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)

    img = imageio.imread(filename, pilmode="L")
    if img.shape != (28, 28):
        return "Image size mismatch " + str(img.shape) + ". \nOnly (28, 28) is acceptable."
    img = img.reshape(784)
    predicted_label = np.argmax(model.predict(np.array([img]))[0], axis=-1)
    print(predicted_label)
    return str(predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
