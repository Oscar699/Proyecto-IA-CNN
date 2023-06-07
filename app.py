import os

from flask import Flask, url_for
from flask import render_template, request, redirect
from keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
from keras.models import load_model
from numpy import argmax, max
from tensorflow import expand_dims, nn
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

import config

# Precision de la red neuronal: 0.7220708727836609
# Se testea con 15 img el modelo de flores y acerta 10 -> 66.6% de precision.
# No aprendio muy bien sobre los dandelion y las deisy

app = Flask(__name__)
app.config.from_object(config.DevelopmentConfig)
redCNN = tf.keras.saving.load_model('model.hdf5', encoding='latin1')
redCNN.summary()
classes = ['un diente de león', 'una margarita', 'un tulipan', 'un girasol', 'una rosa']


def predecir(fileName):
    IMG_SIZE = 150
    path = f'./static/images/{fileName}'
    img = load_img(path=path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)
    predictions = redCNN.predict(img_array)
    score = nn.softmax(predictions[0])


    return f"La imagen se parece a {classes[np.argmax(score)]}"



@app.route('/', methods=["GET", "POST"])
def inicio():
    prediccion = ""
    filename = ""
    if request.method == 'POST':
        file = request.files['imgFile']
        if file.filename != "":
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            prediccion = predecir(filename)

    return render_template("index.html", prediccion=prediccion, file=filename)

if __name__ == '__main__':
    app.run(debug=True)
