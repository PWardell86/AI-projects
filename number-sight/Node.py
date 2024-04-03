import tensorflow as tf
import numpy as np
from tensorflow import keras
from parse_training_data import parse_training_data, parse_line

def train(version=0):
    # Define the model
    model = keras.Sequential([
      keras.layers.Flatten(input_shape=(16, 16)),  # Assuming input data is 28x28 images
      keras.layers.Dense(128, activation="relu"),  # First hidden layer with 128 neurons and ReLU activation
      keras.layers.Dense(10, activation="softmax")  # Output layer with 10 neurons (for 10 class classification)
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model (replace with your training data)
    x_train, y_train = parse_training_data(16, 16, 10)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    model.fit(x_train, y_train, epochs=5)
    model.save("./models/model-0.keras")
    
    return model

if __name__ == "__main__":
    test_three = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0"
    useSaved = True
    if useSaved:
        model = keras.models.load_model("./models/model-0.keras")
    else:
        model = train()
    data = np.array(parse_line(test_three, 16, 16))
    data = tf.expand_dims(data, axis=0)
    prediction = model.predict(data)
    n, max_chance = 0, 0
    for i in range(len(prediction[0])):
        if prediction[0][i] > max_chance:
            n = i
            max_chance = prediction[0][i]
    print(f"Prediction: {n} with {max_chance*100:.2f}% chance")