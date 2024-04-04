from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import logging
import sys

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

app = Flask(__name__)

model = None
fromFile = False
modelFileName = "./model-0.keras"
height = 20
width = 20

try:
    with open(modelFileName, 'r') as file:
        fromFile = True
        model = keras.models.load_model(modelFileName)
except FileNotFoundError:
    fromFile = False
    
if not fromFile:
    # Define the model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(width, height)),  # Assuming input data is 28x28 images
        keras.layers.Dense(205, activation="relu"),  # First hidden layer with 128 neurons and ReLU activation
        keras.layers.Dense(10, activation="softmax")  # Output layer with 10 neurons (for 10 class classification)
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/style.css")
def style():
    return render_template('style.css')

@app.route("/script.js")
def script():
    return render_template('script.js')

@app.route("/getprediction", methods=['POST'])
def get_prediction():
    data = request.get_json()['pixels']
    n, p = predict(data)
    int_p = int(p * 100)
    return json.dumps({"number" : n, "confidence" : int_p}), 200

@app.route("/savetrainingdata", methods=['POST'])
def save_training_data():
    json = request.get_json()
    data = json['trainingData']
    with open("training_data_temp.txt", 'a+') as file:
        file.write(data + "\n")
    return "Data saved", 200

@app.route("/train")
def train():
    with open("training_data_temp.txt", 'r') as file:
        xTrain = []
        yTrain = []
        for line in file.readlines():
            arr = list(line)
            parsed = []
            for r in range(height):
                parsed.append(list(map(int, arr[r * width : (r + 1) * width])))
                
            ydata = convert_answer(int(arr[-2]))
            xTrain.append(parsed)
            yTrain.append(ydata)
    
    with open("training_data.txt", 'a+') as file:
        with open("training_data_temp.txt", 'r+') as temp:
            file.write(temp.read())
            temp.write("")
            
    model.fit(np.array(xTrain), np.array(yTrain), epochs=5, verbose=0)
    model.save(modelFileName)
    return "Model trained", 200

def predict(int_array):
    data = np.array(int_array)
    data = tf.expand_dims(data, axis=0)
    return get_likely_number(model.predict(data, verbose=0))

def convert_answer(answer):
    a = [0 for _ in range(10)]
    a[int(answer)] = 1
    return a

def get_likely_number(prediction):
    n, max_chance = 0, 0
    for i in range(len(prediction[0])):
        if prediction[0][i] > max_chance:
            n = i
            max_chance = prediction[0][i]
    return n, max_chance

if __name__ == '__main__':
    app.run(debug=True)